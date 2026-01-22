import os
import certifi
from dotenv import load_dotenv
from typing import Optional, List, Dict

from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_NAME = "fieldflo_rag"
COLLECTION_NAME = "help_articles"
INDEX_NAME = "vector_index"

def get_vector_store():
    """Get the vector store instance."""
    client = MongoClient(MONGO_DB_URL, tlsCAFile=certifi.where())
    collection = client[DB_NAME][COLLECTION_NAME]
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    return MongoDBAtlasVectorSearch(
        collection=collection,
        embedding=embeddings,
        index_name=INDEX_NAME
    )

def build_filter(topics: Optional[List[str]] = None) -> Dict:
    """
    Build MongoDB MQL Filter.
    This enables Metadata-Filtered RAG pattern.
    """
    if not topics:
        return {}
    
    # "in" operator allows matching ANY of the provided topics
    return {"topics": {"$in": topics}}

def retrieve_documents(query: str, topics: Optional[List[str]] = None, k: int = 4):
    """
    Retrieve documents using vector similarity + metadata filtering.
    """
    vector_store = get_vector_store()
    filters = build_filter(topics)
    
    if filters:
        print(f"🔍 Searching with filters: {filters}")
        results = vector_store.similarity_search(query, k=k, pre_filter=filters)
    else:
        print("🔍 Searching without filters (Naive RAG)")
        results = vector_store.similarity_search(query, k=k)
        
    return results

if __name__ == "__main__":
    # Simple manual test
    docs = retrieve_documents("How do I add a lead?", topics=["CRM"])
    for d in docs:
        print(f"- [{d.metadata['title']}] (Topics: {d.metadata['topics']})")