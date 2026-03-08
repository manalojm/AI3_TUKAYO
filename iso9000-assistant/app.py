import streamlit as st
import numpy as np
import faiss
import tempfile
import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ISO 9000 QMS Assistant",
    page_icon="🏆",
    layout="wide",
)

# ── Styles ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0c10;
    color: #d4d4d4;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #f0f0f0;
    line-height: 1.1;
    letter-spacing: -1px;
}

.hero-accent {
    color: #22c55e;
}

.hero-sub {
    font-size: 0.95rem;
    color: #666;
    margin-top: 8px;
    font-weight: 300;
}

.tag {
    display: inline-block;
    background: #161a24;
    border: 1px solid #22c55e33;
    color: #22c55e;
    font-size: 0.72rem;
    font-family: 'DM Sans', monospace;
    padding: 3px 10px;
    border-radius: 20px;
    margin-right: 6px;
    margin-bottom: 6px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.answer-card {
    background: #111318;
    border: 1px solid #1e2330;
    border-left: 3px solid #22c55e;
    border-radius: 8px;
    padding: 24px 28px;
    margin-top: 12px;
    font-size: 0.95rem;
    line-height: 1.8;
    color: #d4d4d4;
}

.question-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #f0f0f0;
    margin-bottom: 4px;
}

.meta-row {
    font-size: 0.78rem;
    color: #444;
    margin-top: 10px;
    font-family: 'DM Sans', sans-serif;
}

.meta-green { color: #22c55e; }

.status-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
}

.status-ready { background: #0d2818; color: #22c55e; border: 1px solid #22c55e44; }
.status-wait  { background: #1a1500; color: #f59e0b; border: 1px solid #f59e0b44; }

.divider { border-color: #1e2330; margin: 20px 0; }

.stFileUploader > div {
    background: #111318 !important;
    border: 1px dashed #2a2d3a !important;
    border-radius: 8px !important;
}

.stTextInput > div > div > input {
    background-color: #111318 !important;
    border: 1px solid #2a2d3a !important;
    border-radius: 6px !important;
    color: #d4d4d4 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
}

.stButton > button {
    background-color: #22c55e !important;
    color: #0a0c10 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 10px 28px !important;
    letter-spacing: 0.3px !important;
    font-size: 0.9rem !important;
    width: 100% !important;
}

.stButton > button:hover {
    background-color: #16a34a !important;
}

section[data-testid="stSidebar"] {
    background-color: #0d0f14 !important;
    border-right: 1px solid #1e2330 !important;
}

.example-btn {
    background: #111318;
    border: 1px solid #1e2330;
    border-radius: 6px;
    padding: 8px 14px;
    font-size: 0.83rem;
    color: #888;
    margin-bottom: 6px;
    cursor: pointer;
    transition: border-color 0.2s;
}
</style>
""", unsafe_allow_html=True)

# ── Groq client ──────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

SYSTEM_PROMPT = """You are an expert ISO 9000:2005 Quality Management Systems assistant.
Your knowledge comes exclusively from the ISO 9000:2005 standard document provided as context.

Rules:
1. Answer ONLY based on the ISO 9000 context provided — never hallucinate.
2. Cite the specific clause number (e.g. "Clause 3.1.1", "Section 2.4") when available.
3. Keep answers clear, concise, and professional.
4. If the context does not contain the answer, say so explicitly.
5. Do NOT generate exam questions, MCQs, or test-style content.
6. Use bullet points or numbered lists when listing multiple items.
7. Do NOT add any information not explicitly stated in the provided context."""

# ── Cached model loader ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embed_model():
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# ── PDF processing ───────────────────────────────────────────────────────────
def process_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        tmp_path = tmp.name

    reader = PdfReader(tmp_path)
    pages = []
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        pages.append(f"--- Page {i} ---\n{text}")
    full_text = "\n\n".join(pages)
    os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(full_text)

    embed_model = load_embed_model()
    embeddings = embed_model.encode(
        chunks, batch_size=32, show_progress_bar=False,
        convert_to_numpy=True, normalize_embeddings=True,
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return chunks, index, len(reader.pages)

# ── Retrieval ────────────────────────────────────────────────────────────────
def retrieve(question, chunks, index, top_k=6, min_sim=0.15):
    embed_model = load_embed_model()
    q_emb = embed_model.encode(
        [question], convert_to_numpy=True, normalize_embeddings=True
    )
    scores, indices = index.search(q_emb, top_k)
    relevant, sims = [], []
    for idx, score in zip(indices[0], scores[0]):
        if score >= min_sim:
            relevant.append(chunks[idx])
            sims.append(float(score))
    return relevant, sims

# ── Generation ───────────────────────────────────────────────────────────────
def generate(question, context_chunks, api_key):
    if not context_chunks:
        return "I could not find relevant information in the provided document to answer that question."

    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    context = "\n\n".join(
        f"[Context {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content":
                f"Context from ISO 9000:2005:\n\n{context}\n\n"
                f"Question: {question}\n\nAnswer based strictly on the context above."
            }
        ],
        max_tokens=512,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()

# ── Layout ───────────────────────────────────────────────────────────────────
col_main, col_side = st.columns([2, 1], gap="large")

with col_main:
    st.markdown("""
    <div class="hero-title">ISO 9000:2005<br><span class="hero-accent">QMS Assistant</span></div>
    <div class="hero-sub">RAG-powered · Document-grounded · No hallucinations</div>
    <br>
    <span class="tag">Groq LLM</span>
    <span class="tag">FAISS Retrieval</span>
    <span class="tag">ISO 9000:2005</span>
    <span class="tag">CSS181-3</span>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── PDF Upload ────────────────────────────────────────────────────────────
    uploaded_pdf = st.file_uploader(
        "Upload ISO 9000:2005 PDF",
        type=["pdf"],
        help="Upload the ISO 9000:2005 PDF to begin"
    )

    if uploaded_pdf:
        if "chunks" not in st.session_state or st.session_state.get("pdf_name") != uploaded_pdf.name:
            with st.spinner("Processing PDF — building FAISS index..."):
                chunks, index, pages = process_pdf(uploaded_pdf)
                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.pdf_name = uploaded_pdf.name
                st.session_state.pdf_pages = pages
            st.markdown(
                f'<span class="status-pill status-ready">✓ Ready — {len(chunks)} chunks from {pages} pages</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<span class="status-pill status-ready">✓ {st.session_state.pdf_name} loaded — {len(st.session_state.chunks)} chunks</span>',
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            '<span class="status-pill status-wait">⏳ Waiting for PDF upload</span>',
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Question input ────────────────────────────────────────────────────────
    with st.form("ask_form", clear_on_submit=True):
        question = st.text_input(
            "Your question",
            placeholder="e.g. What is the definition of quality according to ISO 9000?",
            label_visibility="collapsed",
            disabled="chunks" not in st.session_state,
        )
        ask_btn = st.form_submit_button(
            "Ask →",
            disabled="chunks" not in st.session_state
        )

    # ── Answer ────────────────────────────────────────────────────────────────
    if ask_btn and question.strip():
        groq_key = st.session_state.get("groq_key", GROQ_API_KEY)
        if not groq_key:
            st.error("Please enter your Groq API key in the sidebar.")
        else:
            with st.spinner("Retrieving context and generating answer..."):
                try:
                    relevant_chunks, scores = retrieve(
                        question,
                        st.session_state.chunks,
                        st.session_state.index
                    )
                    answer = generate(question, relevant_chunks, groq_key)

                    if "history" not in st.session_state:
                        st.session_state.history = []
                    st.session_state.history.insert(0, {
                        "question": question,
                        "answer": answer,
                        "num_chunks": len(scores),
                        "top_score": round(scores[0], 3) if scores else 0.0,
                    })
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── History ───────────────────────────────────────────────────────────────
    if "history" in st.session_state and st.session_state.history:
        st.markdown("<br>", unsafe_allow_html=True)
        for item in st.session_state.history:
            st.markdown(f'<div class="question-label">❓ {item["question"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="answer-card">{item["answer"]}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="meta-row">📌 Grounded on <span class="meta-green">{item["num_chunks"]}</span> sections · '
                f'top similarity: <span class="meta-green">{item["top_score"]}</span></div>',
                unsafe_allow_html=True
            )
            st.markdown('<hr class="divider">', unsafe_allow_html=True)

with col_side:
    st.markdown("<br><br>", unsafe_allow_html=True)

    # ── API Key ───────────────────────────────────────────────────────────────
    st.markdown("#### 🔑 Groq API Key")
    groq_input = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        value=st.session_state.get("groq_key", ""),
        label_visibility="collapsed",
        help="Get a free key at console.groq.com"
    )
    if groq_input:
        st.session_state.groq_key = groq_input

    st.caption("Get a free key at [console.groq.com](https://console.groq.com)")

    st.markdown("---")

    # ── Example questions ─────────────────────────────────────────────────────
    st.markdown("#### 💡 Example Questions")
    examples = [
        "What is the definition of quality?",
        "What are the eight QM principles?",
        "Corrective vs preventive action?",
        "Role of top management in QMS?",
        "How is a process defined?",
        "What types of audits exist?",
        "What is customer satisfaction?",
        "What is a nonconformity?",
    ]
    for ex in examples:
        st.markdown(f'<div class="example-btn">▸ {ex}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── About ─────────────────────────────────────────────────────────────────
    st.markdown("#### ℹ️ About")
    st.markdown("""
**Model**: Llama-3.1-8B via Groq  
**Retriever**: FAISS + all-mpnet-base-v2  
**Chunking**: LangChain RecursiveCharacterTextSplitter  
**Knowledge base**: ISO 9000:2005 (user-uploaded)  
**Project**: CSS181-3 Final Project
    """)
