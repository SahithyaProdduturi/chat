import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
import os

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Functions ---

@st.cache_resource
def get_vectorstore_from_url(url):
    """Load website content and store embeddings persistently."""
    loader = WebBaseLoader(url)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(documents)

    persist_dir = "chroma_db"
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    vector_store = Chroma.from_documents(
        document_chunks,
        embeddings,
        persist_directory=persist_dir
    )
    return vector_store


@st.cache_resource
def get_default_vectorstore():
    """Use default text when no website is given."""
    default_text = """
    Hello! I am your helpful assistant. You can ask me anything about general topics, technology, science, history, or how this chatbot works.
    """

    document = Document(page_content=default_text)
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents([document])

    persist_dir = "default_chroma_db"
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    vector_store = Chroma.from_documents(
        document_chunks,
        embeddings,
        persist_directory=persist_dir
    )
    return vector_store


def get_context_retriever_chain(vector_store):
    """Retrieve context from the stored embeddings."""
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Generate a search query based on the conversation to get relevant information.")
    ])

    return create_history_aware_retriever(llm, retriever, prompt)


def get_conversational_rag_chain(retriever_chain):
    """Generate responses based on retrieved context."""
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer based on the provided context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(user_input):
    """Generate response for user query using RAG pipeline."""
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })

    return response['answer']

# --- Streamlit App ---

st.set_page_config(page_title="Chat with Websites", page_icon="💬")
st.title("Chat with Websites")

# Sidebar for website URL input
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Enter Website URL", key="website_url")

# Initialize session state
if "stored_website_url" not in st.session_state:
    st.session_state.stored_website_url = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hello! How can I assist you?")]

# Load vector store (website or default)
if not website_url:
    if st.session_state.vector_store is None:
        st.session_state.vector_store = get_default_vectorstore()
    st.info("No website URL provided. Using default knowledge base.")
else:
    if website_url != st.session_state.stored_website_url:
        st.session_state.stored_website_url = website_url
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

# Chat UI
st.write("### Chat Interface")

# Show chat history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write("🤖 **Bot:** " + message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write("👤 **You:** " + message.content)

# Get user input
user_query = st.chat_input("Type your message here")

if user_query:
    response = get_response(user_query)

    # Add messages to history
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))

    # Display the latest messages
    with st.chat_message("Human"):
        st.write("👤 **You:** " + user_query)
    with st.chat_message("AI"):
        st.write("🤖 **Bot:** " + response)

