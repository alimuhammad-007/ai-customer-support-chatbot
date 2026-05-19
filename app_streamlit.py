import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Customer Support Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Customer Support Chatbot")

# -----------------------------
# LOAD MODELS (CACHE)
# -----------------------------
@st.cache_resource
def load_models():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile"
    )

    return embeddings, llm

embeddings, llm = load_models()

# -----------------------------
# SESSION STATE
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "db" not in st.session_state:
    st.session_state.db = None

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:

    st.header("📄 Upload PDFs")

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type="pdf",
        accept_multiple_files=True
    )

    if st.button("Process PDFs"):

        if uploaded_files:

            with st.spinner("Processing PDFs..."):

                docs = []

                # save + load PDFs
                for file in uploaded_files:

                    temp_path = f"temp_{file.name}"

                    with open(temp_path, "wb") as f:
                        f.write(file.read())

                    loader = PyPDFLoader(temp_path)
                    docs.extend(loader.load())

                # split text
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )

                chunks = splitter.split_documents(docs)

                # create FAISS db
                db = FAISS.from_documents(chunks, embeddings)

                # save vector db
                db.save_local("vectorstore")

                # load vector db
                st.session_state.db = FAISS.load_local(
                    "vectorstore",
                    embeddings,
                    allow_dangerous_deserialization=True
                )

            st.success("PDFs processed successfully ✔")

        else:
            st.warning("Please upload at least one PDF.")

# -----------------------------
# CHAT FUNCTION
# -----------------------------
def ask_bot(question):

    if st.session_state.db is None:
        return "Please upload and process PDFs first."

    # similarity search
    docs = st.session_state.db.similarity_search(question, k=2)

    # shorter context = faster response
    context = "\n\n".join(
        [doc.page_content[:300] for doc in docs]
    )

    # optimized prompt
    prompt = f"""
You are a helpful AI customer support assistant.

Answer only from the provided context.

Context:
{context}

Question:
{question}

Give a short and clear answer.
"""

    response = llm.invoke(prompt)

    return response.content

# -----------------------------
# SHOW CHAT HISTORY
# -----------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# -----------------------------
# CHAT INPUT
# -----------------------------
user_input = st.chat_input("Ask your question...")

# -----------------------------
# HANDLE USER INPUT
# -----------------------------
if user_input:

    # store user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # assistant response
    with st.chat_message("assistant"):

        message_placeholder = st.empty()

        # loading effect
        with st.spinner("Thinking..."):

            answer = ask_bot(user_input)

        # typing animation
        full_response = ""

        for char in answer:

            full_response += char

            message_placeholder.markdown(full_response + "▌")

            time.sleep(0.005)

        message_placeholder.markdown(full_response)

    # store assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })