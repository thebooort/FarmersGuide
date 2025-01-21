import chromadb

def check_chroma_db(persist_directory="chroma_db"):
    """Check if ChromaDB contains stored documents."""
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(name="research_papers")

    count = collection.count()
    print(f"✅ ChromaDB contains {count} indexed documents.")

    # Retrieve and print a sample document (if available)
    if count > 0:
        sample = collection.peek(1)  # Fetch 1 sample
        print("\n🔹 Sample Retrieved Document:")
        print(f"📝 Content: {sample['documents'][0]}")
        print(f"📄 Source: {sample['metadatas'][0]['source']}")
    else:
        print("❌ No documents found in ChromaDB. Ensure PDFs were processed.")

if __name__ == "__main__":
    check_chroma_db()
