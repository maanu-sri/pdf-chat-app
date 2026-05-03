# 📄 Chat with PDF — RAG Application

A conversational AI app that lets you upload any PDF and ask questions about it using RAG (Retrieval Augmented Generation).

##  Features
- Upload any PDF file
- Ask questions in natural language
- Powered by LLaMA 3.3 via Groq
- Fast semantic search using FAISS
- Clean dark themed UI

##  Tech Stack
- LangChain
- Groq (LLaMA 3.3)
- FAISS Vector Database
- HuggingFace Embeddings
- Streamlit

##  How it works
1. PDF is loaded and split into chunks
2. Chunks are converted to embeddings
3. Stored in FAISS vector database
4. User question is matched to relevant chunks
5. LLM answers based on those chunks

##  Setup
Get your free Groq API key at [console.groq.com](https://console.groq.com)
