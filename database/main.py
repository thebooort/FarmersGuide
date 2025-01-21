from extract_pdfs import extract_text_from_pdfs
from chunk_text import split_text_into_chunks
from embed_store import store_embeddings
from query_rag import query_database

if __name__ == "__main__":
    print("🚀 Starting Full Pipeline...")

    # Extract text from PDFs
    docs = extract_text_from_pdfs()

    # Split text into chunks
    chunked_docs = split_text_into_chunks(docs)

    # Store embeddings in ChromaDB
    store_embeddings(chunked_docs)

    # Query system
    while True:
        user_query = input("\n🔍 Enter query (or 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        query_database(user_query)

    print("✅ Pipeline Complete!")
