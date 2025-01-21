import chromadb
from langchain_community.embeddings import OpenAIEmbeddings

def query_database(query_text, persist_directory="chroma_db", top_k=5):
    """Queries ChromaDB for similar document chunks."""
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(name="research_papers")

    # Initialize OpenAI Embeddings
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")
    query_embedding = embedding_model.embed_query(query_text)

    # Retrieve results
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    if not results["documents"]:
        print("❌ No relevant documents found.")
        return

    print(f"\n🔍 Query: {query_text}\n")
    for idx, (doc, metadata) in enumerate(zip(results["documents"], results["metadatas"])):
        # ✅ Fix: Check if metadata is empty before accessing it
        if metadata and isinstance(metadata, list) and len(metadata) > 0:
            source = metadata[0].get("source", "Unknown Source")
        else:
            source = "Unknown Source"

        print(f"🔹 Result {idx+1}:")
        print(f"📄 Source: {source}")
        print(f"📝 Content: {doc}\n")

if __name__ == "__main__":
    while True:
        user_query = input("\n🔍 Enter query (or 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        query_database(user_query)
