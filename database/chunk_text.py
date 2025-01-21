from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text_into_chunks(documents, chunk_size=1000, chunk_overlap=100):
    """Splits extracted text into chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_docs = []

    for doc in documents:
        chunks = text_splitter.split_text(doc["text"])
        chunked_docs.extend([{"text": chunk, "source": doc["source"]} for chunk in chunks])

    return chunked_docs

if __name__ == "__main__":
    from extract_pdfs import extract_text_from_pdfs
    docs = extract_text_from_pdfs()
    chunked_docs = split_text_into_chunks(docs)
    print(f"✅ Created {len(chunked_docs)} chunks")
