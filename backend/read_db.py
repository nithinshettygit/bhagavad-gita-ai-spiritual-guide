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
from pydantic import BaseModel, Field

# NEW IMPORTS FOR MULTI-QUERY RETRIEVER AND TOOLS
from langchain.retrievers import MultiQueryRetriever
from langchain_core.tools import tool # For defining tools
from langchain.agents import create_tool_calling_agent, AgentExecutor # For creating and executing the agent
from langchain_core.runnables import RunnablePassthrough, RunnableParallel # For creating complex chains
from langchain_core.output_parsers import StrOutputParser # For parsing LLM output to string

import sqlite3
import json
import logging

# Configure logging for LangChain to see agent's thoughts
logging.basicConfig(level=logging.INFO)
logging.getLogger("langchain.agents").setLevel(logging.INFO)
logging.getLogger("langchain.llms").setLevel(logging.INFO)
logging.getLogger("langchain.chains").setLevel(logging.INFO)
logging.getLogger("langchain.retrievers").setLevel(logging.INFO)


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
    description="An AI assistant for the Bhagavad Gita, powered by Groq and RAG, with advanced agentic capabilities.",
    version="1.1.0",
)

# Global variables to store initialized components
faiss_db = None
embeddings_model = None
llm = None
processed_df = None


# --- Tool Definitions ---

@tool
def get_verse_content(chapter_num: int = Field(..., description="The chapter number of the Bhagavad Gita."),
                      verse_num: int = Field(..., description="The verse number within the specified chapter.")) -> str:
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
        return f"Verse not found for Chapter {chapter_num}, Verse {verse_num} in the Bhagavad Gita data. Please verify the chapter and verse numbers and try again."

    content = None
    if 'text' in verse_data.columns:
        content = verse_data['text'].iloc[0]
    elif 'processed_content' in verse_data.columns:
        content = verse_data['processed_content'].iloc[0]

    if content is None:
        return "Error: Verse content column ('text' or 'processed_content') not found in CSV. Please inform the user about this internal issue."
    
    return f"Retrieved content for Chapter {chapter_num}, Verse {verse_num}:\n{content}"


# Simple in-memory glossary for demonstration
BHAGAVAD_GITA_GLOSSARY = {
    "dharma": "Righteous conduct; moral duty; the natural law governing the universe; one's purpose in life.",
    "karma": "Action; the universal law of cause and effect, where every action (physical or mental) creates corresponding reactions.",
    "moksha": "Liberation from the cycle of birth and death (samsara); spiritual liberation; ultimate freedom.",
    "yoga": "Union; various spiritual disciplines aimed at uniting the individual soul (Atman) with the Universal Soul (Brahman). Refers to different paths like Karma Yoga (action), Jnana Yoga (knowledge), Bhakti Yoga (devotion), and Dhyana Yoga (meditation).",
    "atman": "The individual soul or self; the true self beyond identification with the phenomenal world.",
    "brahman": "The ultimate reality; the absolute; the supreme cosmic spirit; the universal consciousness.",
    "samsara": "The cycle of birth, death, and rebirth; the continuous flow of worldly existence.",
    "guna": "Qualities or attributes of nature (Prakriti): Sattva (goodness, purity), Rajas (passion, activity), Tamas (ignorance, inertia). All material existence is composed of these three gunas.",
    "maya": "Illusion; the power by which Brahman creates the phenomenal world; the veil that obscures the true nature of reality.",
    "bhakti": "Devotion; loving adoration of God or a divine form.",
    "jnana": "Knowledge; wisdom; especially spiritual knowledge leading to liberation.",
    "sannyasa": "Renunciation; the path of renouncing worldly attachments and pursuits.",
    "tapas": "Austerity; spiritual discipline involving self-control and penance.",
    "buddhi": "Intellect; discriminating faculty; spiritual discernment."
}

@tool
def get_glossary_definition(term: str = Field(..., description="The Sanskrit term or philosophical concept to define.")) -> str:
    """
    Looks up and retrieves the definition of a specific Sanskrit term or philosophical concept
    related to the Bhagavad Gita.
    Use this tool when the user asks for the meaning of a specific term, e.g., "What is Dharma?"
    or "Define Karma."
    """
    normalized_term = term.lower().strip()
    definition = BHAGAVAD_GITA_GLOSSARY.get(normalized_term)
    if definition:
        return f"Definition of '{term}': {definition}"
    else:
        return f"Definition for '{term}' not found in the glossary. Perhaps you can find more context in the verses."


@tool
def find_related_verses(topic_query: str = Field(..., description="A query describing the topic or concept for which related verses are needed.")) -> str:
    """
    Finds and retrieves verses from the Bhagavad Gita that are related to a given topic or concept.
    This tool performs a semantic search to find verses that discuss the specified topic.
    Use this when the user asks for 'other verses about...', 'passages discussing...', or 'similar teachings on...'.
    """
    global faiss_db # Access the global FAISS database

    if faiss_db is None:
        return "Error: Vector store not loaded. Cannot find related verses."

    # Use the FAISS retriever to find relevant documents (verses)
    # We'll limit to 3 for brevity, but this can be adjusted.
    retrieved_docs = faiss_db.similarity_search(topic_query, k=3) 
    
    if not retrieved_docs:
        return f"No related verses found for the topic: '{topic_query}'."

    verses_info = []
    for doc in retrieved_docs:
        chapter = doc.metadata.get('Chapter', 'N/A')
        verse = doc.metadata.get('Verse', 'N/A')
        content = doc.page_content # Assumes page_content holds the verse text
        verses_info.append(f"Chapter {chapter}, Verse {verse}: \"{content[:150]}...\"") # Truncate for summary

    return "Related verses found:\n" + "\n".join(verses_info)


@tool
def summarize_text(text: str = Field(..., description="The text content to be summarized.")) -> str:
    """
    Summarizes a given piece of text.
    Use this tool when you have a long piece of information (e.g., from retrieved context)
    and need to provide a concise summary to the user or to yourself for internal processing.
    """
    global llm # Access the global LLM

    if llm is None:
        return "Error: Language model not loaded. Cannot perform summarization."

    summarization_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a concise summarization assistant. Summarize the following text briefly and accurately, focusing on the main points."),
        ("human", "{text}")
    ])
    summarization_chain = summarization_prompt | llm | StrOutputParser()

    try:
        summary = summarization_chain.invoke({"text": text})
        return f"Summary of the provided text: {summary}"
    except Exception as e:
        return f"Error during summarization: {e}. Could not summarize the text."


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

            if msg_type == "human":
                messages.append(HumanMessage(content=content, **lc_kwargs_loaded, example=example, name=name))
            elif msg_type == "ai":
                messages.append(AIMessage(content=content, **lc_kwargs_loaded, example=example, tool_calls=tool_calls, name=name))
            elif msg_type == "system":
                messages.append(SystemMessage(content=content, **lc_kwargs_loaded, name=name))
            elif msg_type == "tool":
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
        # print(f"Added message ({msg_type}) for session {self.session_id}: {content[:50]}...") # Too verbose for production logging

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
                model_name="llama3-8b-8192", # Using Llama 3 8B
                temperature=0.5,
                max_tokens=1024 # Increased max_tokens for longer responses/summaries
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
async def get_verse_by_number_direct(chapter_num: int, verse_num: int):
    # This is a direct API endpoint, not using the LLM tool
    # It allows for direct lookup if a UI needs it
    if processed_df is None:
        raise HTTPException(status_code=503, detail="Verse data not loaded. Please wait for server startup or check logs.")

    # Call the tool function directly
    # Note: the tool function returns a string with "Retrieved content for...",
    # for a direct API call, we might want just the raw content.
    # Let's adjust the tool or create a helper for this direct endpoint if needed.
    # For now, let's just use the direct df lookup like before:
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
    global llm, faiss_db # Ensure globals are accessible

    if llm is None or faiss_db is None:
        raise HTTPException(status_code=503, detail="AI or vector store not ready. Check server logs for initialization errors.")

    query = request.query
    user_id = request.user_id 
    print(f"Query received for user '{user_id}': '{query}'")

    # Define all available tools
    tools = [get_verse_content, get_glossary_definition, find_related_verses, summarize_text]

    # Initialize the MultiQueryRetriever
    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=faiss_db.as_retriever(search_kwargs={"k": 5}), # Retrieve more documents for richer context
        llm=llm, # Use your Groq LLM to generate new queries
    )

    # --- Self-Correction/Reflection Chain ---
    # This chain takes the initial answer, query, and context, and provides feedback.
    # It acts as a critic.
    reflection_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are a critical self-reflection assistant.
        Your task is to review an AI's initial answer to a user's question, given the original query and the context/tool outputs used.
        Evaluate the answer for accuracy, completeness, relevance, and adherence to the persona of DivineGuide-Shri Krishna.
        
        Provide constructive feedback. If the answer is perfect, state 'Answer is excellent.'
        If there are issues, clearly state what could be improved or what was missed.
        Focus on how the answer could be more aligned with the Bhagavad Gita's wisdom and the persona.
        
        Original Query: {query}
        Initial Answer: {answer}
        Retrieved Context/Tool Outputs: {context} (This includes documents and outputs from tools if used)
        """),
        ("human", "Please review the initial answer and provide feedback for improvement."),
    ])
    # The reflection chain will just use the base LLM to generate text feedback
    reflection_chain = reflection_prompt | llm | StrOutputParser()

    # --- Main Agent Prompt ---
    # This prompt guides the primary agent that uses RAG and tools, and later will incorporate reflection.
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are DivineGuide-Shri Krishna, a wise, compassionate, and patient spiritual mentor.
        Your mission is to guide the user toward inner peace, wisdom, and understanding of life's deeper spiritual meaning, drawing exclusively from the timeless wisdom of the Bhagavad Gita.

        You have access to the following tools: {tools}

        Carefully consider the user's question, the chat history, and the retrieved context from the Bhagavad Gita.
        
        **Instructions for Tool Use:**
        -   Use `get_verse_content` when the user explicitly asks for a verse by chapter and verse number (e.g., "What does 2.47 say?").
        -   Use `get_glossary_definition` when the user asks for the definition of a specific Sanskrit term (e.g., "What is Dharma?").
        -   Use `find_related_verses` when the user asks for other verses on a topic (e.g., "Are there other verses about renunciation?").
        -   Use `summarize_text` internally if you retrieve very long documents and need to summarize them to fit your thought process or provide a concise overview to the user.

        **Instructions for Answering:**
        -   Combine information from retrieved context and any tool outputs to formulate your answer.
        -   Maintain the wise and compassionate tone of Shri Krishna.
        -   If an answer is not found in the context or via tools, or if the question is outside the scope of spiritual guidance or the Gita, politely state that you cannot answer from the provided information or that the question is beyond your current capacity. Do not make up answers.
        
        Retrieved Context: {context}
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"), # This is where the agent puts its thoughts and tool outputs
    ])

    # Bind tools to the LLM (this LLM will be used by the agent to decide on tool calls)
    llm_with_tools = llm.bind_tools(tools)

    # Create the agent
    agent = create_tool_calling_agent(llm_with_tools, tools, agent_prompt)

    # Create the AgentExecutor. Set verbose=True to see the agent's thought process in logs.
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    # Define the overall chain combining RAG, the agent, and then reflection.
    # The `RunnableParallel` prepares the input for the agent by first getting the RAG context.
    # We pass the original `input` and `chat_history` through to the agent_executor as well.
    main_agent_rag_chain = RunnableParallel(
        context=multiquery_retriever, # This runs first to get relevant docs
        input=RunnablePassthrough(), # Passes the original user query
        chat_history=RunnablePassthrough(), # Passes the chat history
        agent_scratchpad=RunnablePassthrough(), # Passes an empty scratchpad initially for the agent
    ) | agent_executor

    # --- Integrate Reflection after the main agent has produced an answer ---
    # We'll create a step that takes the agent's output and original inputs,
    # then runs the reflection chain.
    # For a simple one-shot reflection, we'll get the reflection feedback and
    # then include it in the final LLM response to show the process, or let the LLM
    # re-evaluate before giving final output.

    # Simpler approach: AgentExecutor gives initial answer. Then reflection happens,
    # and we pass BOTH to a final LLM for ultimate output.
    final_response_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are DivineGuide-Shri Krishna. A user asked a question, and an AI agent provided an initial answer.
        You also have feedback from a self-reflection process.
        Your task is to provide the final, refined answer to the user, incorporating the best aspects of the initial answer and addressing any feedback.
        Ensure your response is wise, compassionate, and exclusively draws from the Bhagavad Gita's teachings.
        
        Original Query: {query}
        Initial AI Answer: {initial_answer}
        Self-Reflection Feedback: {reflection_feedback}
        Retrieved Context (if any was used for the initial answer): {context}
        """),
        ("human", "Based on the above, please provide the most complete and refined answer to the user's original query."),
    ])

    # This chain orchestrates the RAG/Tool agent, then the reflection, then final answer generation.
    full_pipeline = RunnableParallel(
        # First, run the main agent (which includes RAG and tool use)
        agent_output=main_agent_rag_chain,
        # Pass original query, history, and context through for later steps
        original_input=RunnablePassthrough(lambda x: x["input"]), # Get original input from the outer RunnableParallel
        original_chat_history=RunnablePassthrough(lambda x: x["chat_history"]),
        retrieved_context=RunnablePassthrough(lambda x: x["context"]) # This will be the Documents from the retriever
    ).with_config(run_name="Initial_Agent_Run") | RunnableParallel(
        # Now, prepare inputs for reflection
        reflection_feedback=RunnableParallel(
            query=RunnablePassthrough(lambda x: x["original_input"]),
            answer=RunnablePassthrough(lambda x: x["agent_output"]["output"]), # Get the actual answer string
            context=RunnablePassthrough(lambda x: x["retrieved_context"]), # Pass the retrieved context documents
            # intermediate_steps=RunnablePassthrough(lambda x: x["agent_output"]["intermediate_steps"]) # If we want to include agent's internal steps
        ) | reflection_chain, # Run the reflection LLM
        # Pass through initial agent output and original inputs for final answer generation
        initial_answer_from_agent=RunnablePassthrough(lambda x: x["agent_output"]["output"]),
        original_input_for_final=RunnablePassthrough(lambda x: x["original_input"]),
        original_chat_history_for_final=RunnablePassthrough(lambda x: x["original_chat_history"]),
        retrieved_context_for_final=RunnablePassthrough(lambda x: x["retrieved_context"])
    ).with_config(run_name="Reflection_And_Preparation") | RunnableParallel(
        final_answer=final_response_prompt | llm | StrOutputParser(),
        # Pass through intermediate results if you want them in the final FastAPI response
        initial_answer=RunnablePassthrough(lambda x: x["initial_answer_from_agent"]),
        reflection_feedback=RunnablePassthrough(lambda x: x["reflection_feedback"])
    ).with_config(run_name="Final_Answer_Generation")


    with_message_history = RunnableWithMessageHistory(
        full_pipeline, # Use the new comprehensive pipeline
        get_session_history,
        input_messages_key="original_input_for_final", # Key for user input to the entire history-managed chain
        history_messages_key="original_chat_history_for_final", # Key for chat history to the entire history-managed chain
        # All other inputs are handled by the RunnableParallel steps
    )
    
    history_obj = None
    try:
        # The invoke method of RunnableWithMessageHistory now expects just the original input and config
        # The internal chain will handle passing it around.
        result = with_message_history.invoke(
            {"input": query, "chat_history": get_session_history(user_id).messages},
            config={"configurable": {"session_id": user_id}}
        )
        
        history_obj = get_session_history(user_id) 

        # The result of the `full_pipeline` is what we return
        # It will contain 'final_answer', 'initial_answer', 'reflection_feedback'
        ai_answer = result["final_answer"]

        # Optionally, you can log the reflection feedback for debugging
        print(f"\n--- Self-Reflection Feedback for '{query}' ---")
        print(result.get("reflection_feedback", "No reflection feedback generated."))
        print("-------------------------------------------\n")

        return {
            "query": query,
            "answer": ai_answer,
            "initial_answer_before_reflection": result.get("initial_answer"), # Optional: to see the raw agent output
            "reflection_feedback": result.get("reflection_feedback") # Optional: to expose feedback in API
        }
    except Exception as e:
        print(f"Error during agent chain invocation for query '{query}', user '{user_id}': {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process query: {e}. Please check server logs.")
    finally:
        if history_obj and hasattr(history_obj, 'close') and callable(history_obj.close):
            history_obj.close()