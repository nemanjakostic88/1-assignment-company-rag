import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.documents import Document
from pymongo import MongoClient
import certifi

# Load environment variables
load_dotenv()

# Configuration
MONGO_DB_URL = os.getenv("MONGO_DB_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_NAME = "fieldflo_rag"
COLLECTION_NAME = "help_articles"
INDEX_NAME = "vector_index"

def load_json_data(filepath: str) -> List[Dict[str, Any]]:
    """Load the source JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Handle the specific structure: {"data": [...]}
    return data.get("data", [])

def process_timestamp(ts: int) -> str:
    """Convert unix timestamp to readable string."""
    return datetime.fromtimestamp(ts).strftime('%Y-%m-%d')

def create_documents(articles: List[Dict[str, Any]]) -> List[Document]:
    """
    Transform JSON objects into LangChain Documents.
    Strategy: Combine Title + Description + Body for embedding context.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    docs = []
    
    for article in articles:
        # metadata extraction
        metadata = {
            "article_id": str(article.get("id")),
            "title": article.get("title"),
            "url": article.get("url"),
            "topics": article.get("topics", []),
            "updated_at": process_timestamp(article.get("updated_at", 0)),
            "source": "FieldFlo Help Center"
        }
        
        # We combine title and description into the body content 
        # so the chunk has context even if it's in the middle of the article.
        full_content = (
            f"Title: {article.get('title')}\n"
            f"Description: {article.get('description')}\n\n"
            f"{article.get('body')}"
        )
        
        # Split the text
        chunks = text_splitter.create_documents([full_content])
        
        # Assign metadata to every chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata = metadata.copy()
            chunk.metadata["chunk_index"] = i
            docs.append(chunk)
            
    return docs

def setup_mongodb():
    """Initialize DB and clear existing collection."""
    client = MongoClient(MONGO_DB_URL, tlsCAFile=certifi.where())
    db = client[DB_NAME]
    
    if COLLECTION_NAME in db.list_collection_names():
        db[COLLECTION_NAME].delete_many({})
        print(f"Cleared existing documents in {COLLECTION_NAME}")
    else:
        db.create_collection(COLLECTION_NAME)
        print(f"Created collection {COLLECTION_NAME}")
        
    return client, db[COLLECTION_NAME]

def main():
    print("🚀 Starting Ingestion Pipeline...")
    
    # 1. Load Data
    source_file = "help_rag_ready_published_only.json"
    raw_articles = load_json_data(source_file)
    print(f"Loaded {len(raw_articles)} articles from JSON.")
    
    # 2. Chunk and Process
    documents = create_documents(raw_articles)
    print(f"Created {len(documents)} chunks.")
    
    # 3. Store in MongoDB
    client, collection = setup_mongodb()
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    print("Embedding and Storing... (this may take a moment)")
    vector_store = MongoDBAtlasVectorSearch.from_documents(
        documents=documents,
        embedding=embeddings,
        collection=collection,
        index_name=INDEX_NAME
    )
    
    print("✅ Ingestion Complete.")
    
    # Instructions for Index Creation
    print("\n" + "="*60)
    print("IMPORTANT: Create this Search Index in MongoDB Atlas:")
    print("="*60)
    print(f"""
{{
  "fields": [
    {{
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    }},
    {{
      "type": "filter",
      "path": "topics"
    }},
    {{
      "type": "filter",
      "path": "article_id"
    }}
  ]
}}
    """)

if __name__ == "__main__":
    main()