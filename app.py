import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

:root {
    --bg:        #0d1117;
    --surface:   #161b22;
    --border:    #30363d;
    --accent:    #58a6ff;
    --accent2:   #3fb950;
    --text:      #e6edf3;
    --muted:     #8b949e;
    --user-bg:   #1c2128;
    --ai-bg:     #13181f;
    --danger:    #f85149;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Header */
.main-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 24px 0 8px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
}
.main-header .icon {
    font-size: 2.2rem;
    line-height: 1;
}
.main-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: var(--accent);
    margin: 0;
    letter-spacing: -0.5px;
}
.main-header .subtitle {
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 2px;
    font-family: 'IBM Plex Mono', monospace;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

.user-bubble {
    background: var(--user-bg);
    border: 1px solid var(--border);
    border-radius: 12px 12px 2px 12px;
    padding: 14px 18px;
    margin: 8px 0 8px 60px;
    font-size: 0.95rem;
    line-height: 1.6;
}

.ai-bubble {
    background: var(--ai-bg);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 2px 12px 12px 12px;
    padding: 16px 20px;
    margin: 8px 60px 8px 0;
    font-size: 0.93rem;
    line-height: 1.7;
}

.source-tag {
    display: inline-block;
    background: rgba(88,166,255,0.1);
    border: 1px solid rgba(88,166,255,0.3);
    color: var(--accent);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 20px;
    margin: 6px 4px 0 0;
}

.sources-row {
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid var(--border);
}

/* Chat input */
[data-testid="stChatInput"] textarea {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    border-radius: 10px !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(88,166,255,0.15) !important;
}

/* Sidebar book pills */
.book-pill {
    background: rgba(63,185,80,0.08);
    border: 1px solid rgba(63,185,80,0.2);
    border-radius: 6px;
    padding: 6px 10px;
    margin: 4px 0;
    font-size: 0.75rem;
    color: var(--accent2);
    font-family: 'IBM Plex Mono', monospace;
    word-break: break-word;
}

/* Status badge */
.status-online {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(63,185,80,0.1);
    border: 1px solid rgba(63,185,80,0.3);
    color: var(--accent2);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
}
.dot { width: 7px; height: 7px; border-radius: 50%; background: var(--accent2);
       animation: pulse 2s infinite; display: inline-block; }
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}

/* Buttons */
.stButton button {
    background: var(--surface) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    transition: all 0.2s;
}
.stButton button:hover {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* Scrollbar */
::-webkit-scrollbar       { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; max-width: 900px; }
</style>
""", unsafe_allow_html=True)

# ── LOAD RESOURCES ───────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models...")
def load_resources():
    embed = SentenceTransformer("all-MiniLM-L6-v2")
    qdrant = QdrantClient(
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_API_KEY"]
    )
    groq = Groq(api_key=st.secrets["GROQ_API_KEY"])
    return embed, qdrant, groq

embed_model, qdrant_client, groq_client = load_resources()

# ── GET BOOK LIST ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_book_list():
    try:
        results = qdrant_client.scroll(
            collection_name="medical_books",
            limit=500,
            with_payload=["source"]
        )
        books = set()
        for point in results[0]:
            if point.payload and "source" in point.payload:
                books.add(point.payload["source"])
        return sorted(books)
    except:
        return []

# ── QUERY FUNCTION ───────────────────────────────────────────
def ask(question, history):
    # Embed question
    q_vec = embed_model.encode([question]).tolist()[0]

    # Retrieve from Qdrant
    hits = qdrant_client.query_points(
        collection_name="medical_books",
        query=q_vec,
        limit=10,
        with_payload=True
    ).points

    context_parts, sources = [], set()
    for hit in hits:
        context_parts.append(f"[{hit.payload['source']}]\n{hit.payload['text']}")
        sources.add(hit.payload['source'].replace(".pdf",""))

    context = "\n\n---\n\n".join(context_parts)

    # Build messages
    messages = [{
        "role": "system",
        "content": f"""You are a senior surgical consultant and medical educator. You have mastered the content of multiple surgical textbooks. You answer like you are teaching a junior doctor — precise, authoritative, and clinically focused.

Rules:
- Answer strictly from the textbook context provided
- If multiple books cover the topic, synthesize them
- Use correct medical terminology
- Be direct and confident — avoid unnecessary hedging
- Structure answers clearly with numbered points
- End with the most important clinical pearl

Available textbooks: {', '.join(sources) if sources else 'Medical textbook library'}"""
    }]

    # Add conversation history (last 4 turns)
    for h in history[-4:]:
        messages.append({"role": "user",      "content": h["q"]})
        messages.append({"role": "assistant", "content": h["a"]})

    messages.append({
        "role": "user",
        "content": f"""TEXTBOOK CONTEXT:
{context}

QUESTION: {question}

Answer as a senior surgical consultant:
1. Core Answer (direct, 2–3 sentences)
2. Detailed Explanation
3. Clinical Pearls
4. Source(s): {', '.join(sources)}"""
    })

    # Stream from Groq
    stream = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=0.2,
        max_tokens=1500,
        stream=True
    )

    return stream, sources

# ── SESSION STATE ─────────────────────────────────────────────
if "messages"     not in st.session_state: st.session_state.messages     = []
if "history_pairs" not in st.session_state: st.session_state.history_pairs = []

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="margin-bottom:16px">', unsafe_allow_html=True)
    st.markdown('<span class="status-online"><span class="dot"></span>System Online</span>',
                unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### 📚 Knowledge Base")
    books = get_book_list()
    if books:
        for book in books:
            st.markdown(f'<div class="book-pill">📖 {book}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="color:var(--muted);font-size:0.75rem;margin-top:10px">'
                    f'{len(books)} books loaded</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:var(--muted);font-size:0.8rem">No books found</div>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💡 Sample Questions")
    samples = [
        "What is the management of acute appendicitis?",
        "Explain the blood supply of the stomach",
        "Types of hernias and their repair",
        "Complications of thyroid surgery",
        "Management of small bowel obstruction",
    ]
    for sample in samples:
        if st.button(sample, key=sample, use_container_width=True):
            st.session_state["prefill"] = sample

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages      = []
        st.session_state.history_pairs = []
        st.rerun()

    st.markdown(
        '<div style="color:var(--muted);font-size:0.7rem;margin-top:16px;line-height:1.5">'
        'Built with Streamlit · Qdrant · Groq<br>'
        'For educational use only</div>',
        unsafe_allow_html=True
    )

# ── MAIN AREA ─────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="icon">🏥</div>
    <div>
        <h1>Medical AI Assistant</h1>
        <div class="subtitle">Ask clinical questions · Powered by your surgical textbook library</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Render chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">🧑‍⚕️ &nbsp; {msg["content"]}</div>',
                    unsafe_allow_html=True)
    else:
        sources_html = ""
        if msg.get("sources"):
            tags = "".join(f'<span class="source-tag">📖 {s}</span>'
                           for s in msg["sources"])
            sources_html = f'<div class="sources-row">{tags}</div>'
        st.markdown(
            f'<div class="ai-bubble">{msg["content"]}{sources_html}</div>',
            unsafe_allow_html=True
        )

# Handle sidebar sample question click
prefill = st.session_state.pop("prefill", None)

# Chat input
prompt = st.chat_input("Ask a clinical question about surgery, anatomy, or medicine...")
if prefill:
    prompt = prefill

if prompt:
    # Show user message
    st.markdown(f'<div class="user-bubble">🧑‍⚕️ &nbsp; {prompt}</div>',
                unsafe_allow_html=True)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Stream AI response
    with st.spinner(""):
        stream, sources = ask(prompt, st.session_state.history_pairs)

    response_placeholder = st.empty()
    full_response = ""

    with response_placeholder.container():
        full_response = st.write_stream(
            chunk.choices[0].delta.content or ""
            for chunk in stream
            if chunk.choices[0].delta.content
        )

    # Show final bubble with sources
    response_placeholder.empty()
    sources_html = ""
    if sources:
        tags = "".join(f'<span class="source-tag">📖 {s}</span>' for s in sources)
        sources_html = f'<div class="sources-row">{tags}</div>'
    st.markdown(
        f'<div class="ai-bubble">{full_response}{sources_html}</div>',
        unsafe_allow_html=True
    )

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": list(sources)
    })
    st.session_state.history_pairs.append({"q": prompt, "a": full_response})
