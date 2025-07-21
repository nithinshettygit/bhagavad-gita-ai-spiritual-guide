import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import torch

# Load environment variables
load_dotenv()

# Define paths (must match rag_utils.py)
FAISS_INDEX_PATH = "faiss_index"

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

def load_vector_store(embeddings, faiss_path: str = FAISS_INDEX_PATH):
    """Loads an existing FAISS vector store."""
    print(f"Loading FAISS index from {faiss_path}...")
    if not os.path.exists(faiss_path):
        raise FileNotFoundError(f"FAISS index not found at {faiss_path}. Please run `python rag_utils.py` first.")
    # Set allow_dangerous_deserialization=True as this is a local, known source
    db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    print("FAISS index loaded.")
    return db

if __name__ == "__main__":
    print("--- Starting RAG Test Retrieval ---")

    # 1. Create embeddings model (must be the same one used for indexing)
    embeddings_model = create_embeddings_model()

    # 2. Load the FAISS vector store
    vector_store = load_vector_store(embeddings_model)

    # 3. Perform a similarity search
    query = "What is the nature of the Self?"
    print(f"\nSearching for documents related to: '{query}'")
    # Retrieve top 3 most relevant documents
    retrieved_docs = vector_store.similarity_search(query, k=1)

    print("\n--- Retrieved Documents (Top 3) ---")
    for i, doc in enumerate(retrieved_docs):
        print(f"--- Document {i+1} ---")
        print(f"Metadata: {doc.metadata}")
        print(f"Content (first 500 chars):\n{doc.page_content[:500]}...")
        print("-" * 30)

    # You can try other queries as well:
    # query_2 = "Describe the three modes of material nature."
    # retrieved_docs_2 = vector_store.similarity_search(query_2, k=2)
    # print(f"\nSearching for documents related to: '{query_2}'")
    # for i, doc in enumerate(retrieved_docs_2):
    #     print(f"--- Document {i+1} ---")
    #     print(f"Metadata: {doc.metadata}")
    #     print(f"Content (first 500 chars):\n{doc.page_content[:500]}...")
    #     print("-" * 30)

    print("--- RAG Test Retrieval Complete ---")