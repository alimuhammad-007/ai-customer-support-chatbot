import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Customer Support Chatbot",
    page_icon="🤖",
    layout="centered"
)

# -----------------------------
# ENVIRONMENT VARIABLES & SECRETS
# -----------------------------
# Use st.secrets for Streamlit Cloud deployment, or os.environ for local.
def get_api_key():
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")

GROQ_API_KEY = get_api_key()

# -----------------------------
# LOAD MODELS (CACHE)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    """Load embeddings and LLM once for performance."""
    if not GROQ_API_KEY:
        st.error("🚨 GROQ_API_KEY is missing. Add it to environment variables or Streamlit secrets.")
        st.stop()
        
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0.2, # Lower temperature for more factual answers
        max_tokens=1024
    )

    return embeddings, llm

embeddings, llm = load_models()

# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------
def process_pdfs(uploaded_files):
    """Safely process uploaded PDFs and create vector store."""
    docs = []
    
    for file in uploaded_files:
        # Secure file handling with tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.read())
            temp_path = temp_file.name

        try:
            loader = PyPDFLoader(temp_path)
            docs.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading {file.name}: {str(e)}")
        finally:
            # Always clean up temp files
            if os.path.exists(temp_path):
                os.remove(temp_path)

    if not docs:
        st.warning("No text could be extracted from the uploaded PDFs.")
        return None

    # Optimized Chunking Strategy for better context retrieval
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    # Create FAISS database in-memory (no need to save local unless requested)
    db = FAISS.from_documents(chunks, embeddings)
    return db

def get_rag_chain(db):
    """Create a highly optimized RAG chain using LangChain Expression Language (LCEL)."""
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4} # Fetch top 4 most relevant chunks
    )

    template = """
You are a highly professional, helpful, and concise AI Customer Support Assistant.
Answer the user's question accurately based ONLY on the provided context.
If the answer is not contained in the context, politely inform the user that you do not have that information.
Do not make up information or guess.

Context:
{context}

Question:
{question}

Answer:"""
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Modern LCEL chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# -----------------------------
# UI & MAIN APP
# -----------------------------
st.title("🤖 AI Customer Support Chatbot")
st.markdown("Upload your company PDFs and ask questions instantly!")

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "db" not in st.session_state:
    st.session_state.db = None

# Sidebar
with st.sidebar:
    st.header("📄 Document Upload")
    st.markdown("Upload knowledge base files for the AI.")
    
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("Train AI", use_container_width=True, type="primary"):
        if uploaded_files:
            with st.spinner("Analyzing and training on PDFs..."):
                db = process_pdfs(uploaded_files)
                if db:
                    st.session_state.db = db
                    st.success("Training complete! The AI is ready. ✔")
        else:
            st.warning("Please upload at least one PDF.")
            
    if st.session_state.db is not None:
        st.success("✅ Database loaded and ready.")
        if st.button("Clear Memory", use_container_width=True):
            st.session_state.db = None
            st.session_state.messages = []
            st.rerun()

# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("How can I help you today?")

if user_input:
    # Append & display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Guard clause if DB isn't ready
    if st.session_state.db is None:
        with st.chat_message("assistant"):
            warning_msg = "Please upload and train the AI on PDFs first using the sidebar."
            st.warning(warning_msg)
            st.session_state.messages.append({"role": "assistant", "content": warning_msg})
        st.stop()

    # Generate & stream response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            rag_chain = get_rag_chain(st.session_state.db)
            
            # Stream response for fast UX
            for chunk in rag_chain.stream(user_input):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
                
            message_placeholder.markdown(full_response)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            full_response = "Sorry, I encountered an error while processing your request."

    # Save assistant message
    if full_response:
        st.session_state.messages.append({"role": "assistant", "content": full_response})