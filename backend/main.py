import os
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
import pandas as pd
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
import torch

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
    description="An AI assistant for the Bhagavad Gita, powered by RAG.",
    version="1.0.0",
)

# Global variables to store loaded resources
faiss_db = None
embeddings_model = None
llm = None
processed_df = None # To store the processed DataFrame for direct lookup

# --- Functions to Load Resources (Copied/Adapted from rag_utils.py) ---

def create_embeddings_model():
    model_name = "BAAI/bge-small-en-v1.5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

def load_vector_store(embeddings, faiss_path: str = FAISS_INDEX_PATH):
    if not os.path.exists(faiss_path):
        raise FileNotFoundError(f"FAISS index not found at {faiss_path}. Please run `python rag_utils.py` first.")
    db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    return db

# --- Startup Event Handler ---
@app.on_event("startup")
async def startup_event():
    """Load the FAISS index, embeddings model, and LLM when the FastAPI app starts."""
    global faiss_db, embeddings_model, llm, processed_df

    print("Loading resources for FastAPI app...")
    
    # Load processed CSV for direct lookup (Option 1)
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise FileNotFoundError(f"Processed CSV not found at {PROCESSED_DATA_PATH}. Please run `python rag_utils.py` first to generate it.")
    processed_df = pd.read_csv(PROCESSED_DATA_PATH, encoding='utf-8')
    print(f"Loaded {len(processed_df)} processed verses into memory.")

    # Create embeddings model
    embeddings_model = create_embeddings_model()
    print("Embeddings model loaded.")

    # Load FAISS vector store
    faiss_db = load_vector_store(embeddings_model)
    print("FAISS vector store loaded.")

    # Initialize LLM using HuggingFaceHub for open-source models
# Ensure HUGGINGFACEHUB_API_TOKEN is set in your .env file
    if os.getenv("HUGGINGFACEHUB_API_TOKEN"):
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2", # Keep this
            task="text-generation", # Keep this
            temperature=0.5,
            max_length=512,
        )
        print(f"HuggingFaceEndpoint LLM initialized with model: mistralai/Mistral-7B-Instruct-v0.2")
    else:
        print("HUGGINGFACEHUB_API_TOKEN not found. LLM for RAG will not be available. Please set it in .env.")
        llm = None

# --- API Endpoints ---

@app.get("/")
async def read_root():
    return {"message": "Welcome to Bhagavad Gita AI Spiritual Guide API. Use /docs for API documentation."}

@app.get("/verse/{chapter_num}/{verse_num}")
async def get_verse_by_number(chapter_num: int, verse_num: int):
    """
    Retrieve a specific verse and its explanation by chapter and verse number.
    """
    if processed_df is None:
        raise HTTPException(status_code=503, detail="Data not loaded yet. Please wait for startup or check logs.")

    verse_data = processed_df[(processed_df['Chapter'] == chapter_num) & (processed_df['Verse'] == verse_num)]

    if verse_data.empty:
        raise HTTPException(status_code=404, detail=f"Verse {chapter_num}.{verse_num} not found.")

    # Assuming 'processed_content' holds the combined text
    return {
        "chapter": chapter_num,
        "verse": verse_num,
        "content": verse_data['processed_content'].iloc[0]
    }

@app.post("/ask")
async def ask_krishna_ai(query: str):
    """
    Ask a question to the Krishna AI, powered by RAG.
    """
    if llm is None or faiss_db is None:
        raise HTTPException(status_code=503, detail="AI model or vector store not loaded. Check API key or startup logs.")

    print(f"Received query: '{query}'")

    # Define the prompt template for the LLM
    # This prompt guides the LLM on how to use the retrieved context
    prompt = ChatPromptTemplate.from_template("""
    You are an AI assistant that provides answers based on the Bhagavad Gita.
    Use only the provided context to answer the question. If the answer is not in the context,
    state that you don't know, but encourage the user to ask another question about the Gita.
    Ensure your answer is concise, accurate, and directly addresses the question based on the Gita's teachings.

    Context:
    {context}

    Question: {input}
    """)

    # Create the RAG chain
    # StuffDocumentsChain puts all retrieved documents directly into the prompt context
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(faiss_db.as_retriever(), document_chain)

    # Invoke the chain to get the answer
    response = retrieval_chain.invoke({"input": query})
    
    # The response object will contain 'answer' and 'context' (the retrieved documents)
    print(f"Generated answer for query: '{query}'")
    return {"query": query, "answer": response["answer"]}