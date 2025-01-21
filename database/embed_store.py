import chromadb
from langchain_community.embeddings import OpenAIEmbeddings
from tqdm import tqdm

def store_embeddings(chunked_documents, persist_directory="chroma_db"):
    """Stores chunked text embeddings in ChromaDB, preventing duplicates."""
    
    # ✅ Use PersistentClient to persist data across restarts
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(name="research_papers")

    # Get existing indexed sources
    existing_docs = collection.peek(1000)  # Fetch up to 1000 docs
    existing_sources = set()
    
    if existing_docs and "metadatas" in existing_docs:
        for metadata in existing_docs["metadatas"]:
            if isinstance(metadata, list) and len(metadata) > 0:
                existing_sources.add(metadata[0].get("source", ""))  # Extract source names

    print(f"🔍 Existing Papers in DB: {existing_sources}")

    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

    # Process only new documents
    new_chunks = [doc for doc in chunked_documents if doc["source"] not in existing_sources]

    if not new_chunks:
        print("✅ No new documents found. Skipping embedding step.")
        return

    print(f"🆕 Adding {len(new_chunks)} new documents to ChromaDB...")

    for i in tqdm(range(len(new_chunks)), desc="Indexing New Chunks"):
        chunk = new_chunks[i]
        embedding = embedding_model.embed_query(chunk["text"])
        collection.add(
            ids=[f"chunk_{i}"],
            documents=[chunk["text"]],
            metadatas=[{"source": chunk["source"]}],
            embeddings=[embedding]
        )

    # ✅ Explicitly persist changes
    # client.persist()
    print("✅ New documents stored in ChromaDB!")

if __name__ == "__main__":
    from chunk_text import split_text_into_chunks
    from extract_pdfs import extract_text_from_pdfs
    docs = extract_text_from_pdfs()
    chunked_docs = split_text_into_chunks(docs)
    store_embeddings(chunked_docs)
