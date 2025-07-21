import os
from dotenv import load_dotenv
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import torch
import re # Import regex for potential future cleaning

# Load environment variables from .env file
load_dotenv()

# --- Updated Data Paths ---
ORIGINAL_DATA_FILE = "Bhagwad_Gita.csv"
PROCESSED_DATA_FILE = "bhagavad_gita_processed.csv" # New intermediate file
DATA_DIR = "data"
ORIGINAL_DATA_PATH = os.path.join(DATA_DIR, ORIGINAL_DATA_FILE)
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, PROCESSED_DATA_FILE)
FAISS_INDEX_PATH = "faiss_index"

# Ensure the data directory exists
os.makedirs(DATA_DIR, exist_ok=True)


def preprocess_and_save_data(original_file_path: str, processed_file_path: str):
    """
    Reads the original CSV, preprocesses relevant columns, and saves to a new CSV.
    """
    if not os.path.exists(original_file_path):
        raise FileNotFoundError(f"Original CSV file not found at {original_file_path}")

    print(f"Loading original data from {original_file_path} for preprocessing...")
    df = pd.read_csv(original_file_path, encoding='utf-8')

    processed_data = []
    for index, row in df.iterrows():
        chapter = row['Chapter']
        verse = row['Verse']
        transliteration = str(row['Transliteration']).strip() if pd.notna(row['Transliteration']) else ""
        eng_meaning = str(row['EngMeaning']).strip() if pd.notna(row['EngMeaning']) else ""
        word_meaning = str(row['WordMeaning']).strip() if pd.notna(row['WordMeaning']) else ""

        # Combine the relevant information into a single content string
        # You can adjust this format as needed
        content = (
            f"Chapter {chapter}, Verse {verse}:\n"
            f"Sanskrit Transliteration: {transliteration}\n"
            f"English Meaning: {eng_meaning}\n"
            f"Word Meaning: {word_meaning}"
        )
        
        # --- Optional: Add more cleaning steps here if needed ---
        # Example: Remove extra whitespace, newlines, or specific patterns
        content = re.sub(r'\s+', ' ', content).strip() # Replace multiple whitespaces with single space
        content = content.replace('\n', ' ') # Replace newlines with spaces for a single line string in CSV

        processed_data.append({
            "ID": row['ID'],
            "Chapter": chapter,
            "Verse": verse,
            "processed_content": content # This column holds the combined text
        })

    processed_df = pd.DataFrame(processed_data)
    processed_df.to_csv(processed_file_path, index=False, encoding='utf-8')
    print(f"Processed data saved to {processed_file_path}")
    return processed_df # Return the processed DataFrame for direct use


def convert_df_to_documents(df: pd.DataFrame) -> list[Document]:
    """
    Converts a pandas DataFrame (with 'processed_content' column)
    into a list of LangChain Document objects.
    """
    documents = []
    print("Converting DataFrame to LangChain Documents...")
    for index, row in df.iterrows():
        # Use the 'processed_content' directly
        page_content = row['processed_content']

        metadata = {
            "source": f"Bhagavad Gita - Chapter {row['Chapter']}, Verse {row['Verse']}",
            "chapter": row['Chapter'],
            "verse": row['Verse'],
            "id": row['ID']
        }
        documents.append(Document(page_content=page_content, metadata=metadata))
    print(f"Created {len(documents)} LangChain Documents.")
    return documents


def split_documents(documents: list[Document], chunk_size=1000, chunk_overlap=200):
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks

def create_embeddings_model():
    """Creates and returns a HuggingFace embeddings model."""
    model_name = "BAAI/bge-small-en-v1.5"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    print(f"Using HuggingFace Embeddings model: {model_name} on device: {device}")
    return embeddings

def create_vector_store(chunks: list[Document], embeddings, faiss_path: str = FAISS_INDEX_PATH):
    """Creates a FAISS vector store from document chunks and embeddings."""
    print("Creating FAISS vector store...")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(faiss_path)
    print(f"FAISS index saved to {faiss_path}")
    return db

def load_vector_store(embeddings, faiss_path: str = FAISS_INDEX_PATH):
    """Loads an existing FAISS vector store."""
    print(f"Loading FAISS index from {faiss_path}...")
    if not os.path.exists(faiss_path):
        raise FileNotFoundError(f"FAISS index not found at {faiss_path}. Please run `python rag_utils.py` first.")
    db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    print("FAISS index loaded.")
    return db

if __name__ == "__main__":
    print("--- Starting FAISS Index Generation Workflow ---")

    # Step 1: Preprocess the original data and save an intermediate CSV
    # Check if processed CSV already exists to skip reprocessing
    if not os.path.exists(PROCESSED_DATA_PATH):
        processed_df = preprocess_and_save_data(ORIGINAL_DATA_PATH, PROCESSED_DATA_PATH)
    else:
        print(f"Processed data CSV already exists at {PROCESSED_DATA_PATH}. Loading it directly.")
        processed_df = pd.read_csv(PROCESSED_DATA_PATH, encoding='utf-8')
    
    # Step 2: Convert the processed DataFrame into LangChain Document objects
    documents = convert_df_to_documents(processed_df)

    # Step 3: Split documents into chunks
    chunks = split_documents(documents)

    # Step 4: Create embeddings model
    embeddings_model = create_embeddings_model()

    # Step 5: Create and save FAISS vector store
    faiss_db = create_vector_store(chunks, embeddings_model)
    print("--- FAISS Index Generation Complete ---")