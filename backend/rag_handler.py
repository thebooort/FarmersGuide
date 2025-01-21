import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Ensure API key is loaded
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY is missing. Check your .env file.")

# Initialize LangChain’s ChatOpenAI (GPT-4o Mini)
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    openai_api_key=OPENAI_API_KEY
)

# System message for AI context
SYSTEM_MESSAGE = SystemMessage(
    content="""You are an AI-powered farming assistant, providing expert advice on agricultural practices based on scientific research.
    Use reliable sources, retrieved documents, and best practices to answer user queries.
    If unsure, acknowledge uncertainty rather than generating incorrect information.
    """
)

# Initialize ChromaDB for RAG retrieval
embedding_function = OpenAIEmbeddings()
vector_store = Chroma(persist_directory="./backend/chroma_db", embedding_function=embedding_function)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

def generate_response(user_query: str, chat_history: list = None) -> dict:
    """
    Generates a response using GPT-4o Mini, incorporating relevant RAG documents.

    Args:
        user_query (str): User's question.
        chat_history (list): Previous chat messages for context (optional).

    Returns:
        dict: AI response in JSON format.
    """
    messages = [SYSTEM_MESSAGE]

    # Add chat history for context (if available)
    if chat_history:
        for entry in chat_history:
            if entry["role"] == "user":
                messages.append(HumanMessage(content=entry["content"]))
            else:
                messages.append(AIMessage(content=entry["content"]))

    # Retrieve relevant documents from ChromaDB
    retrieved_docs = retriever.invoke(user_query)
    retrieved_texts = "\n".join([doc.page_content for doc in retrieved_docs])

    # Append retrieved context to user query
    augmented_query = f"""
    --- Relevant Research Papers ---
    {retrieved_texts}
    ------------------------------
    User Question: {user_query}
    """

    messages.append(HumanMessage(content=augmented_query))

    try:
        # Invoke GPT-4o Mini
        response = llm.invoke(messages)
        bot_response = response.content.strip() if response.content else "No response from AI."

        return {"response": bot_response}  # ✅ Always return JSON
    except Exception as e:
        return {"error": f"❌ Error calling GPT-4o: {str(e)}"}  # ✅ Always return JSON
