import os
from pathlib import Path
from chromadb import PersistentClient
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY is missing. Check your .env file.")

# ChromaDB path
CHROMA_DB_PATH = Path(__file__).resolve().parent / "../database/chroma_db"

# Initialize ChromaDB client
try:
    client = PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_or_create_collection(name="research_papers")
    print(f"✅ Successfully loaded ChromaDB from: {CHROMA_DB_PATH}")
except Exception as e:
    print(f"❌ Failed to load ChromaDB: {str(e)}")
    exit()

# Initialize OpenAI embeddings
embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)

# Check document count
try:
    doc_count = collection.count()
    print(f"✅ ChromaDB contains {doc_count} indexed documents.")
except Exception as e:
    print(f"❌ Failed to count documents: {str(e)}")

# Test query
TEST_QUERY = "coffee"

try:
    # Generate embedding for the query
    query_embedding = embedding_function.embed_query(TEST_QUERY)
    
    # Query ChromaDB
    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    if not results["documents"]:
        print("❌ No documents retrieved.")
    else:
        print(f"✅ Retrieved {len(results['documents'])} document(s):")
        for i, (doc, metadata) in enumerate(zip(results["documents"], results["metadatas"]), 1):
            source = metadata[0].get("source", "Unknown Source") if metadata else "Unknown Source"
            print(f"\nDocument {i}:")
            print(f"📄 Source: {source}")
            print(f"📝 Content: {doc[:300]}...")  # Display first 300 characters of content
except Exception as e:
    print(f"❌ Error during retrieval: {str(e)}")
