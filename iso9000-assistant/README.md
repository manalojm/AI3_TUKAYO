# 🏆 ISO 9000:2005 QMS Assistant

A RAG-powered chatbot grounded entirely in the ISO 9000:2005 Quality Management Systems standard.

**CSS181-3 Final Project**

## Stack
- **LLM**: Llama-3.1-8B via Groq API (free)
- **Retriever**: FAISS + sentence-transformers/all-mpnet-base-v2
- **Chunking**: LangChain RecursiveCharacterTextSplitter
- **Frontend**: Streamlit
- **Knowledge base**: ISO 9000:2005 (user-uploaded PDF)

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How to deploy on Streamlit Community Cloud

1. Fork or push this repo to GitHub
2. Go to share.streamlit.io
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Add your Groq API key as a secret: `GROQ_API_KEY = "gsk_..."`
6. Deploy!

## Usage

1. Upload the ISO 9000:2005 PDF
2. Enter your Groq API key in the sidebar (or set it as a secret)
3. Ask any question about the standard
