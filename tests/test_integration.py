import pytest
from app import get_pdf_text, get_text_chunks, get_vector_store
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
def get_vector_store(text_chunks):
    print("Creating vector store from text chunks.")
    try:
        embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        print("Vector store saved locally as 'faiss_index'.")
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

