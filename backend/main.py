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
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage # Added SystemMessage for completeness
from pydantic import BaseModel

# NEW IMPORT: For SQLite database interaction
import sqlite3
import json # To store message content (HumanMessage/AIMessage objects)

# NEW IMPORT: For LCEL chain modification
from langchain_core.runnables import RunnableLambda

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
PROCESSED_DATA_FILE = "bhagavad_gita_processed.csv"
DATA_DIR = "data"
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, PROCESSED_DATA_FILE)
FAISS_INDEX_PATH = "faiss_index"
# NEW CONFIGURATION: SQLite Database Path
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

# NEW CLASS: SQLite-backed chat history implementation
class SQLiteChatMessageHistory:
    """
    A chat message history that stores messages in an SQLite database.
    Designed to be compatible with RunnableWithMessageHistory.
    """
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.conn = self._get_connection()
        self._create_table()

    def _get_connection(self):
        """Establishes and returns a database connection."""
        # Check if the database file exists, if not, it will be created.
        # FIX: Add check_same_thread=False to resolve "SQLite objects created in a thread can only be used in that same thread."
        conn = sqlite3.connect(CHAT_HISTORY_DB, check_same_thread=False)
        conn.row_factory = sqlite3.Row # Allows accessing columns by name
        return conn

    def _create_table(self):
        """Creates the messages table if it does not exist."""
        with self.conn: # Use 'with' statement for automatic commit/rollback
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    type TEXT NOT NULL, -- 'human', 'ai', 'system', 'tool'
                    content TEXT NOT NULL,
                    lc_kwargs TEXT, -- JSON string of kwargs for BaseMessage (e.g., additional_kwargs)
                    example INTEGER, -- 0 or 1, whether it's an example message
                    tool_calls TEXT, -- JSON string of tool calls (for AIMessage)
                    tool_call_id TEXT, -- tool call ID (for ToolMessage, HumanMessage tool_response)
                    name TEXT, -- name of the message sender
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            """)

    @property
    def messages(self) -> list[BaseMessage]:
        """Retrieves all messages for this session from the database."""
        cursor = self.conn.execute(
            "SELECT type, content, lc_kwargs, example, tool_calls, tool_call_id, name FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
            (self.session_id,)
        )
        messages = []
        for row in cursor.fetchall():
            msg_type = row["type"]
            content = row["content"]
            
            # FIX: Defensively load lc_kwargs, defaulting to empty dict if None or invalid JSON
            lc_kwargs_loaded = {}
            if row["lc_kwargs"]:
                try:
                    lc_kwargs_loaded = json.loads(row["lc_kwargs"])
                except json.JSONDecodeError:
                    pass # Keep as empty dict if decode fails

            example = bool(row["example"]) if row["example"] is not None else False
            
            # FIX FOR AIMessage tool_calls: Initialize as empty list, then try to load
            tool_calls = [] # Initialize as empty list (required by AIMessage validation)
            if row["tool_calls"]:
                try:
                    loaded_tool_calls = json.loads(row["tool_calls"])
                    if loaded_tool_calls is not None: # Check if it's not JSON 'null'
                        tool_calls = loaded_tool_calls
                except json.JSONDecodeError:
                    pass # Keep as empty list if decode fails

            tool_call_id = row["tool_call_id"]
            name = row["name"]

            # Reconstruct BaseMessage objects based on type
            if msg_type == "human":
                messages.append(HumanMessage(content=content, **lc_kwargs_loaded, example=example, name=name))
            elif msg_type == "ai":
                messages.append(AIMessage(content=content, **lc_kwargs_loaded, example=example, tool_calls=tool_calls, name=name))
            elif msg_type == "system":
                messages.append(SystemMessage(content=content, **lc_kwargs_loaded, name=name))
            # Add other types if needed (e.g., ToolMessage, FunctionMessage)
            # For simplicity, we are handling basic Human and AI messages.
            # If you add ToolMessages, ensure `tool_call_id` is passed correctly.
        return messages

    def add_message(self, message: BaseMessage):
        """Adds any BaseMessage object to the database."""
        msg_type = message.type
        content = message.content
        
        # FIX: Use getattr to safely access attributes, providing default empty values if not present.
        # This addresses 'AttributeError: lc_kwargs'
        lc_kwargs_dict = getattr(message, 'lc_kwargs', {}) 
        lc_kwargs = json.dumps(lc_kwargs_dict) if lc_kwargs_dict else None # Store as JSON string or None

        example = 1 if getattr(message, 'example', False) else 0 
        
        tool_calls_to_store = None
        if hasattr(message, 'tool_calls'): # Check if the attribute exists
            actual_tool_calls = getattr(message, 'tool_calls', None)
            # If it's not None (could be an empty list [] or a list of tools), dump it
            if actual_tool_calls is not None: 
                tool_calls_to_store = json.dumps(actual_tool_calls)
        
        tool_call_id = getattr(message, 'tool_call_id', None)
        name = getattr(message, 'name', None)

        with self.conn:
            self.conn.execute(
                "INSERT INTO messages (session_id, type, content, lc_kwargs, example, tool_calls, tool_call_id, name) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (self.session_id, msg_type, content, lc_kwargs, example, tool_calls_to_store, tool_call_id, name)
            )
        print(f"Added message ({msg_type}) for session {self.session_id}: {content[:50]}...") # Log for debugging

    # FIX: Implement the add_messages (plural) method as required by LangChain's BaseChatMessageHistory
    def add_messages(self, messages: list[BaseMessage]):
        """Adds a list of BaseMessage objects to the database."""
        for message in messages:
            self.add_message(message)
        # print(f"Added {len(messages)} messages to session {self.session_id} via add_messages.") # Can uncomment for more verbose logging


    def add_user_message(self, message: str):
        """Adds a new user message to the database."""
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str):
        """Adds a new AI message to the database."""
        self.add_message(AIMessage(content=message))

    def clear(self):
        """Clears all messages for this session from the database."""
        with self.conn:
            self.conn.execute("DELETE FROM messages WHERE session_id = ?", (self.session_id,))
        print(f"Chat history cleared for session {self.session_id}.")

    def close(self):
        """Closes the database connection."""
        self.conn.close()
        print(f"Database connection closed for session {self.session_id}.")


# MODIFIED: Retrieves or creates a SQLite-backed chat history for a given session ID
def get_session_history(session_id: str) -> SQLiteChatMessageHistory:
    """
    Retrieves the SQLite-backed chat history for a given session ID.
    """
    # Each call to get_session_history will create a new connection.
    # For a simple app, it's fine. For high-concurrency, a connection pool would be better.
    return SQLiteChatMessageHistory(session_id)


# --- Load Embedding Model ---
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

# --- Load FAISS Vector Store ---
def load_vector_store(embeddings, faiss_path: str = FAISS_INDEX_PATH):
    if not os.path.exists(faiss_path):
        raise FileNotFoundError(
            f"FAISS index not found at '{faiss_path}'. "
            "Please ensure you have run your data processing/indexing script (e.g., `rag_utils.py`) "
            "to create the index and the processed CSV file before starting the server."
        )
    db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    return db

# --- FastAPI Startup Event ---
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

# NEW: FastAPI Shutdown Event (relying on connection closing in finally block per request)
@app.on_event("shutdown")
async def shutdown_event():
    """Placeholder for global shutdown tasks. Connections are mostly per-request."""
    print("\n--- Shutting down: Cleaning up resources ---")
    # For SQLite, connections are managed per request in `get_session_history` and closed in `finally`.
    # No global connection pool to explicitly close here unless you implement one.
    print("Application shutdown complete.")


# --- API Endpoints ---

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

# Pydantic model for the /ask request body
class QueryRequest(BaseModel):
    query: str
    user_id: str # User ID is now crucial for persistent session history

@app.post("/ask")
async def ask_krishna_ai(request: QueryRequest):
    if llm is None or faiss_db is None:
        raise HTTPException(status_code=503, detail="AI or vector store not ready. Check server logs for initialization errors.")

    query = request.query
    user_id = request.user_id 
    print(f"Query received for user '{user_id}': '{query}'")

    # Define the RAG prompt template with chat_history
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", """
            You are DivineGuide-Shri Krishna, a wise, compassionate, and patient spiritual mentor.
            Your mission is to guide the user toward inner peace, wisdom, and understanding of life's deeper spiritual meaning, drawing exclusively from the timeless wisdom of the Bhagavad Gita.

            Beyond just answering questions, you aim to:
            - Provide deep spiritual meaning, not just religious or literal interpretations.
            - Offer comfort, clarity, and practical guidance based on Gita principles.
            - Adapt to the user's emotional tone and learning progress (future feature).
            - Always remain respectful, encouraging, and centered on the Gita's teachings.

            Answer the user's question *only* based on the provided context from the Bhagavad Gita. If the answer is not found in the context, or if the question is outside the scope of spiritual guidance or the Gita, politely state that you cannot answer from the provided information in the context, or that the question is beyond your current capacity as DivineGuide-Shri Krishna. Do not make up answers.
            Ensure your responses resonate with the wisdom and tone of the Bhagavad Gita.
            """),
            # MessagesPlaceholder is essential for injecting history
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Question: {input}"),
            ("system", "Context: {context}"),
        ]
    )

    # Combine retrieved documents with the prompt and LLM
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Create the full retrieval-augmented generation chain
    retrieval_chain = create_retrieval_chain(faiss_db.as_retriever(), document_chain)

    # FIX: Ensure 'output' key is always present in the chain's final output for LangChain's internal tracers.
    # The retrieval_chain returns {'input', 'context', 'answer'}. We add 'output' key here.
    final_chain = retrieval_chain | RunnableLambda(lambda x: {**x, "output": x["answer"]})

    # Wrap the final_chain with RunnableWithMessageHistory
    # This runnable automatically manages the chat history using get_session_history
    # 'input_messages_key' maps to the 'input' in your prompt
    # 'history_messages_key' maps to the 'chat_history' in your prompt (MessagesPlaceholder)
    with_message_history = RunnableWithMessageHistory(
        final_chain, # Use the modified chain here
        get_session_history, # Function to get/create custom history for a session
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    # Temporary variable to hold the history object for manual closing if needed
    history_obj = None 
    try:
        # Invoke the with_message_history runnable
        # The 'config' argument is where you pass the session_id
        result = with_message_history.invoke(
            {"input": query}, # Pass only the current input query
            config={"configurable": {"session_id": user_id}} # Pass the user_id as session_id
        )
        
        # Get the history object that was used in this invocation to potentially close its connection
        # This is a bit advanced and often handled by connection pooling in larger apps.
        # For simplicity in this example, it's illustrative.
        history_obj = get_session_history(user_id) 

        # The 'answer' key is now guaranteed to be present from the chain
        ai_answer = result["answer"]
        
        return {
            "query": query,
            "answer": ai_answer
        }
    except Exception as e:
        print(f"Error during RAG chain invocation for query '{query}', user '{user_id}': {e}")
        # Log the full exception traceback for better debugging if this re-occurs
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process query: {e}. Please check server logs.")
    finally:
        # Ensure the database connection for this specific history object is closed
        if history_obj and hasattr(history_obj, 'close') and callable(history_obj.close):
            history_obj.close()