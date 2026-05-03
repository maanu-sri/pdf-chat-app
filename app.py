import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import tempfile
import os

st.set_page_config(
    page_title="Chat with PDF",
    page_icon="📄",
    layout="centered"
)

st.markdown("""
<style>
    .stApp { background-color: #0f0f0f; }
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .subtitle {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
    }
    .chat-message-user {
        background: #1e1e2e;
        border-left: 3px solid #6366f1;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        color: #fff;
    }
    .chat-message-ai {
        background: #1a1a2e;
        border-left: 3px solid #8b5cf6;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
        color: #e2e2e2;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📄 Chat with your PDF</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload any PDF and ask questions about it</div>', unsafe_allow_html=True)

groq_api_key = st.sidebar.text_input("🔑 Groq API Key", type="password")
st.sidebar.markdown("Get free key at [console.groq.com](https://console.groq.com)")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file and groq_api_key:
    with st.spinner("📖 Reading and processing your PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(uploaded_file.read())
            tmp_path = f.name

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(pages)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)

        os.environ["GROQ_API_KEY"] = groq_api_key
        llm = ChatGroq(model="llama-3.3-70b-versatile")

        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant. Answer based only on the context below.
        Be clear, concise and helpful.
        Context: {context}
        Question: {input}
        """)
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        st.success(f"✅ PDF processed! {len(pages)} pages, {len(chunks)} chunks ready.")
        st.session_state.chain = retrieval_chain

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message-ai">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

    question = st.chat_input("Ask anything about your PDF...")
    if question and "chain" in st.session_state:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("🤔 Thinking..."):
            result = st.session_state.chain.invoke({"input": question})
            answer = result["answer"]
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

elif not groq_api_key:
    st.info("Enter your Groq API key in the sidebar to get started!")
elif not uploaded_file:
    st.info("Upload a PDF file to get started!")
