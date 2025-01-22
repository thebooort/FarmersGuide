import os
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from chromadb import PersistentClient
from langchain_community.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY is missing. Check your .env file.")

# Initialize LangChain’s ChatOpenAI (GPT-4o Mini)
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    openai_api_key= OPENAI_API_KEY
)

# System message for AI context
SYSTEM_MESSAGE = SystemMessage(
    content="""You are an AI-powered farming assistant, providing expert advice on agricultural practices based on scientific research.
    Use reliable sources, retrieved documents, and best practices to answer user queries.
    If unsure, acknowledge uncertainty rather than generating incorrect information. Do not format the answer, use plain text.
    Return two paragraphs at maximum, and avoid long-winded responses.
    """
)

# ChromaDB Path
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "../database/chroma_db")

# Initialize ChromaDB client
try:
    client = PersistentClient(path=CHROMA_DB_PATH)
    collection = client.get_or_create_collection(name="research_papers")
    print(f"✅ Successfully connected to ChromaDB at: {CHROMA_DB_PATH}")
except Exception as e:
    raise RuntimeError(f"❌ Failed to connect to ChromaDB: {str(e)}")

# Initialize OpenAI Embeddings
embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=OPENAI_API_KEY)

# Load CSV data for re-ranking
CSV_PATH = os.path.join(os.path.dirname(__file__), "../data/data_with_answers.csv")
try:
    df = pd.read_csv(CSV_PATH)
    print(f"✅ Loaded CSV dataset from: {CSV_PATH}")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load CSV dataset: {str(e)}")

# Initialize SentenceTransformer for re-ranking
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to compute similarity
def compute_similarity(row, reference_dict, weights, model):
    similarity_scores = []
    total_weight = sum(weights.values())

    for key in reference_dict.keys():
        if key in row:
            embedding1 = model.encode(str(row[key]), convert_to_tensor=True)
            embedding2 = model.encode(str(reference_dict[key]), convert_to_tensor=True)

            # Compute cosine similarity
            cosine_sim = cosine_similarity(
                embedding1.detach().numpy().reshape(1, -1),
                embedding2.detach().numpy().reshape(1, -1)
            )[0][0]

            # Apply weight
            weighted_similarity = weights[key] * cosine_sim
            similarity_scores.append(weighted_similarity)

    return sum(similarity_scores) / total_weight if similarity_scores else 0

# Re-rank retrieved documents
def rerank_documents(retrieved_docs, survey_context, weights, model, top_n=3):
    """
    Re-ranks retrieved documents based on survey context similarity.
    Returns the top N most relevant documents.
    """
    if not retrieved_docs:
        return []

    try:
        # Convert documents to DataFrame for easy processing
        df_retrieved = pd.DataFrame([
            {'content': doc, **doc.metadata} if hasattr(doc, 'metadata') else {'content': doc}
            for doc in retrieved_docs
        ])
    except Exception as e:
        print(f"❌ Error processing retrieved documents: {str(e)}")
        return []

    # Compute similarity scores
    df_retrieved['similarity'] = df_retrieved.apply(
        compute_similarity,
        axis=1,
        reference_dict=survey_context,
        weights=weights,
        model=model
    )

    # Sort and return top documents
    top_docs = df_retrieved.sort_values(by='similarity', ascending=False).head(top_n)

    if top_docs.empty:
        return []

    return top_docs['content'].tolist()  # Return list of document texts

# Main response generation function
def generate_response(user_query: str, chat_history: list = None, survey_context: dict = None) -> dict:
    """
    Generates a response using GPT-4o Mini, incorporating relevant RAG documents and survey context.

    Args:
        user_query (str): User's question, including context.
        chat_history (list): Previous chat messages for context (optional).
        survey_context (dict): Survey data from the user (optional).

    Returns:
        dict: AI response in JSON format.
    """
    messages = [SYSTEM_MESSAGE]

    # Add chat history for context (if available)
    if chat_history:
        for entry in chat_history:
            messages.append(HumanMessage(content=entry["content"]) if entry["role"] == "user" else AIMessage(content=entry["content"]))

    # Retrieve documents from ChromaDB
    retrieved_texts = []
    try:
        query_embedding = embedding_function.embed_query(user_query)
        results = collection.query(query_embeddings=[query_embedding], n_results=10)  # Retrieve more docs for ranking
        retrieved_docs = results["documents"] if "documents" in results else []

        # Flatten retrieved documents to strings
        retrieved_texts = ["\n".join(doc) if isinstance(doc, list) else doc for doc in retrieved_docs] if retrieved_docs else []
    except Exception as e:
        print(f"❌ Retrieval error: {str(e)}")
        retrieved_texts = []

    # Log retrieved documents
    print("\n🔍 Retrieved Documents:")
    for idx, doc in enumerate(retrieved_texts, 1):
        print(f"🔹 Document {idx}: {doc[:200]}...")

    # Define weights for ranking
    weights = {
        'Location': 2,
        'Crop': 3,
        'Ecosystem': 3,
        'Agriculture': 1
    }

    # Re-rank documents
    reranked_docs = rerank_documents(retrieved_docs, survey_context, weights, sentence_model, top_n=3)

    # Prepare retrieved documents for LLM
    # retrieved_texts = "\n".join(reranked_docs) if reranked_docs else "No relevant documents found."

    # Handle both strings and lists from ChromaDB (same as original code)
    retrieved_texts = (
        "\n".join(["\n".join(doc) if isinstance(doc, list) else doc for doc in reranked_docs])
        if reranked_docs
        else "No relevant documents found."
    )

    # Add survey context to the augmented query
    survey_context_str = "\n".join([f"- {k}: {v}" for k, v in survey_context.items()]) if survey_context else "No survey data available."

    # Append retrieved context and survey context to user query
    augmented_query = f"""
    --- Relevant Research Papers ---
    {retrieved_texts}
    ------------------------------
    {survey_context_str}
    ------------------------------
    User Question: {user_query}
    """

    messages.append(HumanMessage(content=augmented_query))

    # Log full context sent to the model
    print("\n🔍 Full Context Sent to LLM:")
    print(augmented_query)

    try:
        # Invoke GPT-4o Mini
        response = llm.invoke(messages)

        if not isinstance(response.content, str):
            bot_response = "No response from AI."
        else:
            bot_response = response.content.strip()

        return {"response": bot_response}  # ✅ Always return JSON
    except Exception as e:
        return {"error": f"❌ Error calling GPT-4o: {str(e)}"}  # ✅ Always return JSON
