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
from langchain_core.documents import Document
# NEW IMPORTS for user-specific memory
from pydantic import BaseModel # NEW: For defining the request body structure

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
PROCESSED_DATA_FILE = "bhagavad_gita_processed.csv"
DATA_DIR = "data"
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, PROCESSED_DATA_FILE)
FAISS_INDEX_PATH = "faiss_index"

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

# Removed: Global store for user-specific chat histories
# Removed: get_session_history function

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

# NEW Pydantic model for the /ask request body
class QueryRequest(BaseModel):
    query: str
    user_id: str # User ID is still accepted but not used for history in this version

@app.post("/ask")
async def ask_krishna_ai(request: QueryRequest):
    if llm is None or faiss_db is None:
        raise HTTPException(status_code=503, detail="AI or vector store not ready. Check server logs for initialization errors.")

    query = request.query
    user_id = request.user_id # Still receive user_id but it's not used for memory
    print(f"Query received for user '{user_id}': '{query}' (No chat history maintained in this version)")

    # Define the RAG prompt template (WITHOUT chat_history placeholder)
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
            # MessagesPlaceholder(variable_name="chat_history"), # REMOVED for no chat history
            ("human", "Question: {input}"),
            ("system", "Context: {context}"),
        ]
    )

    # Combine retrieved documents with the prompt and LLM
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Create the full retrieval-augmented generation chain
    retrieval_chain = create_retrieval_chain(faiss_db.as_retriever(), document_chain)

    try:
        # Directly invoke the retrieval_chain (no message history wrapper)
        result = retrieval_chain.invoke({"input": query})

        ai_answer = result["answer"]
        
        return {
            "query": query,
            "answer": ai_answer
        }
    except Exception as e:
        print(f"Error during RAG chain invocation for query '{query}', user '{user_id}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {e}. Please check server logs.")