import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
import pandas as pd
import torch
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

# NEW IMPORTS FOR MULTI-QUERY RETRIEVER AND TOOLS
from langchain.retrievers import MultiQueryRetriever
from langchain_core.tools import tool # For defining tools
from langchain.agents import create_tool_calling_agent, AgentExecutor # For creating and executing the agent
from langchain_core.runnables import RunnablePassthrough, RunnableParallel # For creating complex chains


import sqlite3
import json

from langchain_core.runnables import RunnableLambda

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
PROCESSED_DATA_FILE = "bhagavad_gita_processed.csv"
DATA_DIR = "data"
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, PROCESSED_DATA_FILE)
FAISS_INDEX_PATH = "faiss_index"
CHAT_HISTORY_DB = "chat_history.db"


# --- FastAPI App Initialization ---
app = FastAPI(
    title="Bhagavad Gita AI Spiritual Guide",
    description="An AI assistant for the Bhagavad Gita, powered by Groq and RAG.",
    version="1.0.0",
)

# Global variables to store initialized components
faiss_db = None
embeddings_model = None
llm = None
processed_df = None


# --- Tool Definition ---
# This function will be called by the LLM when it decides to use the tool
@tool
def get_verse_content(chapter_num: int, verse_num: int) -> str:
    """
    Retrieves the exact content of a specific verse from the Bhagavad Gita
    given its chapter and verse number.
    Use this tool when the user explicitly asks for the content of a verse
    by its chapter and verse number, e.g., "What does chapter 2, verse 47 say?"
    or "Quote chapter 3, verse 14."
    """
    global processed_df # Access the global DataFrame

    if processed_df is None:
        return "Error: Verse data not loaded. Cannot retrieve verse. Please inform the user that the data is unavailable."

    verse_data = processed_df[
        (processed_df['Chapter'] == chapter_num) &
        (processed_df['Verse'] == verse_num)
    ]

    if verse_data.empty:
        return f"Verse not found for Chapter {chapter_num}, Verse {verse_num} in the Bhagavad Gita data. Please verify the chapter and verse numbers."

    content = None
    if 'text' in verse_data.columns:
        content = verse_data['text'].iloc[0]
    elif 'processed_content' in verse_data.columns:
        content = verse_data['processed_content'].iloc[0]

    if content is None:
        return "Error: Verse content column ('text' or 'processed_content') not found in CSV. Please inform the user about this internal issue."
    
    return f"Retrieved content for Chapter {chapter_num}, Verse {verse_num}: {content}"


# --- SQLite-backed chat history implementation (UNCHANGED) ---
class SQLiteChatMessageHistory:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.conn = self._get_connection()
        self._create_table()

    def _get_connection(self):
        conn = sqlite3.connect(CHAT_HISTORY_DB, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_table(self):
        with self.conn:
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    lc_kwargs TEXT,
                    example INTEGER,
                    tool_calls TEXT,
                    tool_call_id TEXT,
                    name TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            """)

    @property
    def messages(self) -> list[BaseMessage]:
        cursor = self.conn.execute(
            "SELECT type, content, lc_kwargs, example, tool_calls, tool_call_id, name FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
            (self.session_id,)
        )
        messages = []
        for row in cursor.fetchall():
            msg_type = row["type"]
            content = row["content"]
            
            lc_kwargs_loaded = {}
            if row["lc_kwargs"]:
                try:
                    lc_kwargs_loaded = json.loads(row["lc_kwargs"])
                except json.JSONDecodeError:
                    pass

            example = bool(row["example"]) if row["example"] is not None else False
            
            tool_calls = []
            if row["tool_calls"]:
                try:
                    loaded_tool_calls = json.loads(row["tool_calls"])
                    if loaded_tool_calls is not None:
                        tool_calls = loaded_tool_calls
                except json.JSONDecodeError:
                    pass

            tool_call_id = row["tool_call_id"]
            name = row["name"]

            # Handle ToolMessage type during loading
            if msg_type == "human":
                messages.append(HumanMessage(content=content, **lc_kwargs_loaded, example=example, name=name))
            elif msg_type == "ai":
                messages.append(AIMessage(content=content, **lc_kwargs_loaded, example=example, tool_calls=tool_calls, name=name))
            elif msg_type == "system":
                messages.append(SystemMessage(content=content, **lc_kwargs_loaded, name=name))
            elif msg_type == "tool": # New: Handle ToolMessage type
                messages.append(ToolMessage(content=content, tool_call_id=tool_call_id, name=name))
        return messages

    def add_message(self, message: BaseMessage):
        msg_type = message.type
        content = message.content
        
        lc_kwargs_dict = getattr(message, 'lc_kwargs', {}) 
        lc_kwargs = json.dumps(lc_kwargs_dict) if lc_kwargs_dict else None

        example = 1 if getattr(message, 'example', False) else 0 
        
        tool_calls_to_store = None
        if hasattr(message, 'tool_calls'):
            actual_tool_calls = getattr(message, 'tool_calls', None)
            if actual_tool_calls is not None: 
                tool_calls_to_store = json.dumps(actual_tool_calls)
        
        tool_call_id = getattr(message, 'tool_call_id', None)
        name = getattr(message, 'name', None)

        with self.conn:
            self.conn.execute(
                "INSERT INTO messages (session_id, type, content, lc_kwargs, example, tool_calls, tool_call_id, name) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (self.session_id, msg_type, content, lc_kwargs, example, tool_calls_to_store, tool_call_id, name)
            )
        print(f"Added message ({msg_type}) for session {self.session_id}: {content[:50]}...")

    def add_messages(self, messages: list[BaseMessage]):
        for message in messages:
            self.add_message(message)

    def add_user_message(self, message: str):
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str):
        self.add_message(AIMessage(content=message))

    def clear(self):
        with self.conn:
            self.conn.execute("DELETE FROM messages WHERE session_id = ?", (self.session_id,))
        print(f"Chat history cleared for session {self.session_id}.")

    def close(self):
        self.conn.close()
        print(f"Database connection closed for session {self.session_id}.")


def get_session_history(session_id: str) -> SQLiteChatMessageHistory:
    return SQLiteChatMessageHistory(session_id)


# --- Load Embedding Model (UNCHANGED) ---
def create_embeddings_model():
    model_name = "BAAI/bge-small-en-v1.5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing embeddings model '{model_name}' on device: {device}")
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

# --- Load FAISS Vector Store (UNCHANGED) ---
def load_vector_store(embeddings, faiss_path: str = FAISS_INDEX_PATH):
    if not os.path.exists(faiss_path):
        raise FileNotFoundError(
            f"FAISS index not found at '{faiss_path}'. "
            "Please ensure you have run your data processing/indexing script (e.g., `rag_utils.py`) "
            "to create the index and the processed CSV file before starting the server."
        )
    db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    return db

# --- FastAPI Startup Event (MODIFIED for processed_df only) ---
@app.on_event("startup")
async def startup_event():
    global faiss_db, embeddings_model, llm, processed_df

    print("\n--- Loading resources for Bhagavad Gita AI Spiritual Guide ---")

    # 1. Load processed verses from CSV
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(
            f"Processed CSV not found at '{PROCESSED_DATA_PATH}'. "
            "Ensure 'data' directory exists in your backend folder and 'bhagavad_gita_processed.csv' is inside it. "
            "You might need to run a data processing script first to create this file."
        )
    try:
        processed_df = pd.read_csv(PROCESSED_DATA_PATH, encoding='utf-8')
        print(f"Loaded {len(processed_df)} verses from processed CSV.")
    except Exception as e:
        print(f"Error loading processed CSV from {PROCESSED_DATA_PATH}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load Bhagavad Gita data: {e}")

    # 2. Initialize Embeddings Model and Load FAISS Vector Store
    try:
        embeddings_model = create_embeddings_model()
        print("Embeddings model initialized.")
        faiss_db = load_vector_store(embeddings_model)
        print("FAISS vector store loaded successfully.")

    except FileNotFoundError as e:
        print(f"FAISS/Data setup error: {e}")
        raise HTTPException(status_code=500, detail=f"Essential RAG components (FAISS index or data) not found: {e}")
    except Exception as e:
        print(f"Error during embeddings or FAISS initialization: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize RAG components: {e}")

    # 3. Initialize Groq LLM
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        try:
            llm = ChatGroq(
                api_key=groq_api_key,
                model_name="llama3-8b-8192",
                temperature=0.5,
                max_tokens=512
            )
            print(f"Groq LLM initialized with model: {llm.model_name}")
        except Exception as e:
            print(f"Error initializing Groq LLM: {e}")
            llm = None
            raise HTTPException(status_code=500, detail=f"Failed to initialize Groq LLM: {e}. Check GROQ_API_KEY and network.")
    else:
        raise EnvironmentError(
            "GROQ_API_KEY not found in environment variables. "
            "Please set it in your .env file. LLM will not be available."
        )
    print("--- All resources loaded successfully ---")


@app.on_event("shutdown")
async def shutdown_event():
    print("\n--- Shutting down: Cleaning up resources ---")
    print("Application shutdown complete.")


@app.get("/")
async def read_root():
    return {"message": "Welcome to Bhagavad Gita AI Spiritual Guide. Use /docs to explore the API."}

@app.get("/verse/{chapter_num}/{verse_num}")
async def get_verse_by_number(chapter_num: int, verse_num: int):
    if processed_df is None:
        raise HTTPException(status_code=503, detail="Verse data not loaded. Please wait for server startup or check logs.")

    verse_data = processed_df[
        (processed_df['Chapter'] == chapter_num) &
        (processed_df['Verse'] == verse_num)
    ]

    if verse_data.empty:
        raise HTTPException(status_code=404, detail=f"Verse not found for Chapter {chapter_num}, Verse {verse_num}.")

    content = None
    if 'text' in verse_data.columns:
        content = verse_data['text'].iloc[0]
    elif 'processed_content' in verse_data.columns:
        content = verse_data['processed_content'].iloc[0]
    
    if content is None:
        raise HTTPException(status_code=500, detail="Verse content column ('text' or 'processed_content') not found in CSV.")

    return {
        "chapter": chapter_num,
        "verse": verse_num,
        "content": content
    }

class QueryRequest(BaseModel):
    query: str
    user_id: str

@app.post("/ask")
async def ask_krishna_ai(request: QueryRequest):
    if llm is None or faiss_db is None:
        raise HTTPException(status_code=503, detail="AI or vector store not ready. Check server logs for initialization errors.")

    query = request.query
    user_id = request.user_id 
    print(f"Query received for user '{user_id}': '{query}'")

    # Define the tools available to the agent
    tools = [get_verse_content]

    # Initialize the MultiQueryRetriever
    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=faiss_db.as_retriever(), # Use your existing FAISS retriever as the base
        llm=llm, # Use your Groq LLM to generate new queries
    )

    # Define the agent's prompt
    # This prompt is designed for a tool-calling agent and will receive context from the retriever
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are DivineGuide-Shri Krishna, a wise, compassionate, and patient spiritual mentor.
        Your mission is to guide the user toward inner peace, wisdom, and understanding of life's deeper spiritual meaning, drawing exclusively from the timeless wisdom of the Bhagavad Gita.

        You have access to the following tools: {tools}

        When asked for specific verse content by chapter and verse number (e.g., "What does chapter 2, verse 47 say?", "Quote verse 3.14"), you MUST use the `get_verse_content` tool to retrieve the exact text.
        For general questions, first consider the relevant context from the Bhagavad Gita provided below.
        Combine information from the retrieved context and any tool outputs to formulate your answer.
        If the answer is not found in the context or via tools, or if the question is outside the scope of spiritual guidance or the Gita, politely state that you cannot answer from the provided information in the context, or that the question is beyond your current capacity as DivineGuide-Shri Krishna. Do not make up answers.
        Ensure your responses resonate with the wisdom and tone of the Bhagavad Gita.
        
        Retrieved Context: {context}
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"), # This is where the agent puts its thoughts and tool outputs
    ])

    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools(tools)

    # Create the agent
    agent = create_tool_calling_agent(llm_with_tools, tools, agent_prompt)

    # Create the AgentExecutor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    # Define the overall chain combining RAG and the agent
    # This chain will first retrieve documents, then pass them as 'context' to the agent.
    # The agent then decides to use tools or answer based on input, history, scratchpad, and the provided context.
    rag_with_tools_chain = RunnableParallel(
        # Retrieve context first using multiquery retriever
        context=multiquery_retriever,
        # Pass other inputs directly to the agent's input
        input=RunnablePassthrough(),
        chat_history=RunnablePassthrough(), # Ensure chat_history is passed through
        agent_scratchpad=RunnablePassthrough(), # Agent needs scratchpad as well
    ) | agent_executor


    with_message_history = RunnableWithMessageHistory(
        rag_with_tools_chain, # Use the new agent-based chain
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        # agent_scratchpad is handled internally by AgentExecutor, context by RunnableParallel
    )
    
    history_obj = None
    try:
        # We need to provide all inputs that the agent_executor expects through the chain,
        # specifically 'input', 'chat_history', and 'agent_scratchpad' (even if empty initially).
        # The 'context' will be populated by the 'multiquery_retriever' part of RunnableParallel.
        result = with_message_history.invoke(
            {"input": query, "chat_history": get_session_history(user_id).messages, "agent_scratchpad": []},
            config={"configurable": {"session_id": user_id}}
        )
        
        history_obj = get_session_history(user_id) 

        # The result of AgentExecutor is usually directly the final answer
        ai_answer = result["output"] # AgentExecutor returns 'output' key

        return {
            "query": query,
            "answer": ai_answer
        }
    except Exception as e:
        print(f"Error during RAG chain invocation for query '{query}', user '{user_id}': {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process query: {e}. Please check server logs.")
    finally:
        if history_obj and hasattr(history_obj, 'close') and callable(history_obj.close):
            history_obj.close()