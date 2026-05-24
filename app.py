import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq, AuthenticationError
import time

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg:         #0a0e1a;
    --surface:    #111827;
    --surface2:   #1a2236;
    --border:     #1f2d45;
    --border2:    #2d3f5c;
    --accent:     #4f9eff;
    --accent-glow:#4f9eff30;
    --green:      #34d399;
    --green-dim:  #34d39920;
    --yellow:     #fbbf24;
    --red:        #f87171;
    --text:       #e2e8f0;
    --text2:      #94a3b8;
    --text3:      #64748b;
    --user-bg:    #151f30;
    --ai-bg:      #0d1420;
    --fallback-bg:#1a1e2b;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif;
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebarCollapseButton"] { color: var(--text2) !important; }

/* ── HEADER ── */
.app-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 28px 0 20px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 28px;
}
.app-header .logo {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, #1a3a6e, #0d2040);
    border: 1px solid var(--border2);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem;
    box-shadow: 0 0 20px rgba(79,158,255,0.15);
    flex-shrink: 0;
}
.app-header h1 {
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--text);
    margin: 0 0 4px;
    letter-spacing: -0.5px;
}
.app-header .sub {
    font-size: 0.85rem;
    color: var(--text2);
    display: flex;
    align-items: center;
    gap: 8px;
}
.badge-online {
    display: inline-flex; align-items: center; gap: 5px;
    background: var(--green-dim);
    border: 1px solid #34d39940;
    color: var(--green);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 500;
}
.dot-pulse {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--green);
    animation: pulse 2s ease-in-out infinite;
    display: inline-block;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

/* ── WELCOME SCREEN ── */
.welcome-wrap {
    text-align: center;
    padding: 60px 20px 40px;
    max-width: 600px;
    margin: 0 auto;
}
.welcome-icon { font-size: 3.5rem; margin-bottom: 16px; }
.welcome-wrap h2 {
    font-size: 1.4rem; font-weight: 600;
    color: var(--text); margin-bottom: 8px;
}
.welcome-wrap p {
    font-size: 1rem; color: var(--text2);
    line-height: 1.6; margin-bottom: 32px;
}
.welcome-wrap .disclaimer {
    font-size: 0.75rem; color: var(--text3);
    margin-top: 20px;
    border-top: 1px solid var(--border);
    padding-top: 12px;
}
.sample-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    text-align: left;
}
.sample-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
    cursor: pointer;
    transition: all 0.2s;
    font-size: 0.88rem;
    color: var(--text2);
    line-height: 1.4;
}
.sample-card:hover {
    border-color: var(--accent);
    color: var(--text);
    background: var(--surface2);
}
.sample-card .icon { font-size: 1.1rem; margin-bottom: 6px; display: block; }

/* ── CHAT BUBBLES ── */
.msg-row { display: flex; margin: 10px 0; align-items: flex-start; gap: 12px; }
.msg-row.user  { flex-direction: row-reverse; }

.avatar {
    width: 36px; height: 36px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0; margin-top: 2px;
}
.avatar.user-av { background: linear-gradient(135deg,#1e3a5f,#0d2040); border: 1px solid var(--border2); }
.avatar.ai-av   { background: linear-gradient(135deg,#1a3a6e,#0d2040); border: 1px solid var(--border2);
                   box-shadow: 0 0 12px var(--accent-glow); }

.bubble {
    max-width: calc(100% - 56px);
    border-radius: 16px;
    padding: 14px 18px;
    font-size: 1.02rem;
    line-height: 1.75;
}
.bubble.user {
    background: var(--user-bg);
    border: 1px solid var(--border2);
    border-top-right-radius: 4px;
    color: var(--text);
    font-weight: 400;
}
.bubble.ai {
    background: var(--ai-bg);
    border: 1px solid var(--border);
    border-top-left-radius: 4px;
    border-left: 3px solid var(--accent);
    color: var(--text);
}
.bubble.ai.fallback {
    border-left-color: var(--yellow);
    background: var(--fallback-bg);
}

/* Markdown inside ai bubble */
.bubble.ai h1,.bubble.ai h2,.bubble.ai h3 {
    color: var(--accent); font-weight: 600; margin: 16px 0 8px;
    font-size: 1.05rem;
}
.bubble.ai strong { color: var(--text); font-weight: 600; }
.bubble.ai ul, .bubble.ai ol { padding-left: 20px; margin: 8px 0; }
.bubble.ai li { margin: 4px 0; }
.bubble.ai code {
    background: var(--surface2); border: 1px solid var(--border2);
    border-radius: 4px; padding: 1px 6px;
    font-family: 'JetBrains Mono', monospace; font-size: 0.88rem;
}
.bubble.ai blockquote {
    border-left: 3px solid var(--accent); margin: 10px 0;
    padding: 6px 12px; background: var(--surface); border-radius: 0 8px 8px 0;
}

/* Sources */
.sources-bar {
    margin-top: 12px;
    padding-top: 10px;
    border-top: 1px solid var(--border);
    display: flex; flex-wrap: wrap; gap: 6px;
    align-items: center;
}
.src-label { font-size: 0.72rem; color: var(--text3); font-weight: 500; }
.src-tag {
    display: inline-flex; align-items: center; gap: 4px;
    background: rgba(79,158,255,0.08);
    border: 1px solid rgba(79,158,255,0.25);
    color: var(--accent);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    padding: 2px 8px;
    border-radius: 20px;
}

/* ── CHAT INPUT (question bar) ── */
[data-testid="stChatInput"] {
    background: #1e2736 !important;          /* custom darker background */
    border: 1px solid #2d3f5c !important;   /* subtle border */
    border-radius: 14px !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #7b8ca3 !important;
    opacity: 1 !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}

/* ── SIDEBAR (repeated) ── */
.sidebar-section { margin-bottom: 20px; }
.sidebar-title {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 1px;
    color: var(--text3); text-transform: uppercase; margin-bottom: 10px;
}
.sb-btn {
    display: block; width: 100%;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 9px 12px;
    color: var(--text2) !important;
    font-size: 0.85rem;
    cursor: pointer;
    text-align: left;
    margin-bottom: 6px;
    transition: all 0.15s;
    line-height: 1.4;
}
.sb-btn:hover { border-color: var(--accent); color: var(--text) !important; }

/* Buttons */
.stButton button {
    background: var(--surface2) !important;
    color: var(--text2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    transition: all 0.2s;
    padding: 8px 14px !important;
}
.stButton button:hover {
    border-color: var(--accent) !important;
    color: var(--text) !important;
}

/* Feedback and suggestion buttons */
.feedback-btns {
    margin-top: 6px;
    display: flex;
    gap: 8px;
    align-items: center;
}
.feedback-btns .stButton button {
    padding: 4px 10px !important;
    font-size: 0.8rem !important;
}
.suggestion-btns {
    margin-top: 8px;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}
.suggestion-btns .stButton button {
    font-size: 0.78rem !important;
    padding: 4px 12px !important;
    background: var(--surface) !important;
    border-color: var(--border2) !important;
    color: var(--accent) !important;
}

/* Scrollbar */
::-webkit-scrollbar       { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0.5rem !important; max-width: 860px; }

/* Spinner / thinking dots */
.thinking {
    display: flex; align-items: center; gap: 10px;
    color: var(--text2); font-size: 0.9rem; padding: 14px 18px;
}
.thinking-dots span {
    display: inline-block; width: 6px; height: 6px;
    border-radius: 50%; background: var(--accent);
    animation: bounce 1.2s ease-in-out infinite;
    margin: 0 2px;
}
.thinking-dots span:nth-child(2) { animation-delay: .2s; }
.thinking-dots span:nth-child(3) { animation-delay: .4s; }
@keyframes bounce { 0%,80%,100%{transform:translateY(0)} 40%{transform:translateY(-6px)} }
</style>
""", unsafe_allow_html=True)


# ── RESOURCES (with error handling) ──────────────────────────
@st.cache_resource(show_spinner="Loading AI models...")
def load_resources():
    required_secrets = ["QDRANT_URL", "QDRANT_API_KEY", "GROQ_API_KEY"]
    for sec in required_secrets:
        if sec not in st.secrets or not st.secrets[sec]:
            raise ValueError(f"Missing secret: {sec}. Please configure it in Streamlit secrets.")

    embed = SentenceTransformer("all-MiniLM-L6-v2")
    qdrant = QdrantClient(url=st.secrets["QDRANT_URL"], api_key=st.secrets["QDRANT_API_KEY"])
    try:
        # Quick connection test
        Groq(api_key=st.secrets["GROQ_API_KEY"]).models.list()
    except AuthenticationError:
        raise AuthenticationError("Invalid Groq API key. Please check your GROQ_API_KEY secret.")
    except Exception as e:
        raise ConnectionError(f"Could not connect to Groq: {e}")

    groq = Groq(api_key=st.secrets["GROQ_API_KEY"])
    return embed, qdrant, groq

try:
    embed_model, qdrant_client, groq_client = load_resources()
except Exception as e:
    st.error(f"❌ **Configuration error:** {e}")
    st.stop()

@st.cache_data(show_spinner=False, ttl=300)
def get_book_list():
    try:
        results = qdrant_client.scroll(
            collection_name="medical_books", limit=1000, with_payload=["source"]
        )
        books = set()
        for pt in results[0]:
            if pt.payload and "source" in pt.payload:
                books.add(pt.payload["source"].replace(".pdf", ""))
        return sorted(books)
    except:
        return []


# ── ASK ──────────────────────────────────────────────────────
def ask(question, history):
    """Return (stream_generator, sources_set, is_fallback)"""
    q_vec = embed_model.encode([question]).tolist()[0]
    try:
        hits = qdrant_client.search(
            collection_name="medical_books",
            query_vector=q_vec,
            limit=12,
            with_payload=True
        )
    except Exception as e:
        error_msg = f"⚠️ Could not search the textbooks: {e}"
        def error_stream():
            yield error_msg
        return error_stream(), set(), True

    context_parts, sources = [], set()
    for hit in hits:
        src = hit.payload.get("source", "Unknown")
        context_parts.append(f"[{src}]\n{hit.payload['text']}")
        sources.add(src.replace(".pdf", ""))

    if not sources:
        fallback_text = (
            "I couldn't find any relevant information in the textbooks for your query. "
            "This might be because the topic is too specific or not covered in the available books.\n\n"
            "**Suggestions to rephrase:**\n"
            "- Use general medical terms\n"
            "- Ask about symptoms, diagnosis, or treatment\n"
            "- Try one of the sample questions from the sidebar\n\n"
            "Here are some follow‑up ideas you can click:"
        )
        def fallback_stream():
            yield fallback_text
        return fallback_stream(), set(), True

    context = "\n\n---\n\n".join(context_parts)

    messages = [{
        "role": "system",
        "content": f"""You are a senior medical consultant and clinical educator with mastery of multiple medical and surgical textbooks. You answer like a brilliant attending physician teaching a resident — precise, structured, and clinically grounded.

Rules:
- Base your answer strictly on the provided textbook context
- Synthesize information from multiple sources when available
- Use correct medical terminology throughout
- Be confident and direct — no unnecessary hedging
- Format answers with clear headers and bullet points for readability
- Always end with the single most important clinical takeaway

Books in knowledge base: {', '.join(sources) if sources else 'Medical textbook library'}"""
    }]

    for h in history[-6:]:
        messages.append({"role": "user",      "content": h["q"]})
        messages.append({"role": "assistant", "content": h["a"]})

    messages.append({
        "role": "user",
        "content": f"""TEXTBOOK CONTEXT:
{context}

CLINICAL QUESTION: {question}

Answer as a senior medical consultant. Use this structure:
## Core Answer
[Direct 2-3 sentence answer]

## Explanation
[Detailed clinical explanation]

## Key Points
[Bullet points of what matters most]

## ⚡ Clinical Pearl
[Single most important takeaway]

Sources: {', '.join(sources)}"""
    })

    try:
        stream = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.2,
            max_tokens=2000,
            stream=True
        )
        return stream, sources, False
    except AuthenticationError:
        error_msg = "❌ **API key error:** The Groq API key is invalid. Please check your `GROQ_API_KEY` secret."
        def error_stream():
            yield error_msg
        return error_stream(), set(), True
    except Exception as e:
        error_msg = f"⚠️ An error occurred while generating the answer: {str(e)}"
        def error_stream():
            yield error_msg
        return error_stream(), set(), True


# ── SESSION STATE ────────────────────────────────────────────
if "messages"      not in st.session_state: st.session_state.messages      = []
if "history_pairs" not in st.session_state: st.session_state.history_pairs = []
if "prefill"       not in st.session_state: st.session_state.prefill        = None


# ── SIDEBAR ──────────────────────────────────────────────────
SAMPLES = [
    ("🔪", "Management of acute appendicitis"),
    ("🩸", "Blood supply of the stomach"),
    ("🧬", "Pathophysiology of Cushing syndrome"),
    ("💊", "Post-op analgesia principles"),
    ("🫀", "Cardiac risk assessment before surgery"),
    ("🦠", "Surgical site infection prevention"),
    ("🧠", "Glasgow Coma Scale interpretation"),
    ("📋", "Pre-operative workup for elective surgery"),
]

with st.sidebar:
    books = get_book_list()

    st.markdown(f"""
    <div style="padding:16px 0 8px">
        <div style="font-size:1rem;font-weight:700;color:var(--text);margin-bottom:4px">🏥 Medical AI</div>
        <span class="badge-online"><span class="dot-pulse"></span>Online · {len(books)} books</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-title">💡 Sample Questions</div>', unsafe_allow_html=True)
    for icon, sample in SAMPLES:
        if st.button(f"{icon}  {sample}", key=sample, use_container_width=True):
            st.session_state.prefill = sample
            st.rerun()

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages      = []
            st.session_state.history_pairs = []
            st.rerun()
    with col2:
        st.markdown(
            f'<div style="font-size:0.75rem;color:var(--text3);padding-top:8px;text-align:center">'
            f'{len(st.session_state.messages)//2} questions</div>',
            unsafe_allow_html=True
        )

    st.markdown(
        '<div style="color:var(--text3);font-size:0.7rem;margin-top:20px;line-height:1.6">'
        'For educational use only.<br>Not a substitute for clinical judgment.<br>Do not enter personal health information.</div>',
        unsafe_allow_html=True
    )


# ── HEADER ───────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="logo">🏥</div>
    <div>
        <h1>Medical AI Assistant</h1>
        <div class="sub">
            <span>Powered by your textbook library</span>
            <span class="badge-online"><span class="dot-pulse"></span>Online</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── WELCOME SCREEN ───────────────────────────────────────────
def render_welcome():
    st.markdown("""
    <div class="welcome-wrap">
        <div class="welcome-icon">📖</div>
        <h2>What would you like to know?</h2>
        <p>Ask any clinical question. I'll answer based on your medical textbook library with structured, consultant‑level responses.</p>
        <div class="disclaimer">
            Your questions are processed securely. No personal data is stored. This tool is for educational use only and does not replace professional medical advice.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── RENDER MESSAGES ──────────────────────────────────────────
def render_message(role, content, sources=None, is_fallback=False):
    if role == "user":
        st.markdown(f"""
        <div class="msg-row user">
            <div class="avatar user-av">🧑‍⚕️</div>
            <div class="bubble user">{content}</div>
        </div>""", unsafe_allow_html=True)
    else:
        bubble_class = "bubble ai" + (" fallback" if is_fallback else "")
        src_html = ""
        if sources:
            tags = "".join(f'<span class="src-tag">📖 {s}</span>' for s in sources)
            src_html = f'<div class="sources-bar"><span class="src-label">Sources:</span>{tags}</div>'
        st.markdown(f'<div class="msg-row"><div class="avatar ai-av">🤖</div><div class="{bubble_class}">', unsafe_allow_html=True)
        st.markdown(content)
        if src_html:
            st.markdown(src_html, unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)


# ── FEEDBACK & SUGGESTION HELPERS ─────────────────────────────
SUGGESTIONS = [
    ("Explain more", "Explain more details about this topic"),
    ("Risk factors", "What are the risk factors?"),
    ("Treatment", "How is it treated?"),
    ("Complications", "What are the complications?")
]

def set_feedback(idx, val):
    st.session_state.messages[idx]["feedback"] = val

def set_prefill(q):
    st.session_state.prefill = q


# ── DISPLAY MESSAGES ─────────────────────────────────────────
if not st.session_state.messages:
    render_welcome()
else:
    last_assistant_idx = None
    for i in range(len(st.session_state.messages)-1, -1, -1):
        if st.session_state.messages[i]["role"] == "assistant":
            last_assistant_idx = i
            break

    for i, msg in enumerate(st.session_state.messages):
        render_message(
            msg["role"],
            msg["content"],
            msg.get("sources"),
            msg.get("is_fallback", False)
        )

        if msg["role"] == "assistant" and i == last_assistant_idx:
            # Feedback
            if msg.get("feedback") is None:
                col_fb1, col_fb2, _ = st.columns([0.1, 0.1, 0.8])
                with col_fb1:
                    st.button("👍", key=f"fb_up_{i}", help="Helpful",
                              on_click=set_feedback, args=(i, "positive"))
                with col_fb2:
                    st.button("👎", key=f"fb_down_{i}", help="Not helpful",
                              on_click=set_feedback, args=(i, "negative"))
            else:
                st.caption(f"Feedback: {'👍 helpful' if msg['feedback']=='positive' else '👎 not helpful'}")

            # Suggestion buttons
            st.markdown('<div class="suggestion-btns">', unsafe_allow_html=True)
            cols = st.columns(len(SUGGESTIONS))
            for idx, (label, question) in enumerate(SUGGESTIONS):
                with cols[idx]:
                    st.button(label, key=f"sugg_{i}_{idx}",
                              on_click=set_prefill, args=(question,))
            st.markdown('</div>', unsafe_allow_html=True)


# ── CHAT INPUT ───────────────────────────────────────────────
prompt = st.chat_input("Ask a clinical question (e.g. 'Management of acute appendicitis')")
if st.session_state.prefill:
    prompt = st.session_state.prefill
    st.session_state.prefill = None


# ── PROCESS QUESTION ─────────────────────────────────────────
if prompt:
    render_message("user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    thinking_placeholder = st.empty()
    thinking_placeholder.markdown("""
    <div class="thinking">
        <div class="thinking-dots">
            <span></span><span></span><span></span>
        </div>
        Searching textbooks and generating answer...
    </div>
    """, unsafe_allow_html=True)

    stream, sources, is_fallback = ask(prompt, st.session_state.history_pairs)
    thinking_placeholder.empty()

    response_ph = st.empty()
    full_response = ""
    for chunk in stream:
        token = chunk.choices[0].delta.content if hasattr(chunk, 'choices') else chunk
        if token is None:
            continue
        full_response += token
        response_ph.markdown(full_response + "▌")
    response_ph.empty()

    render_message("assistant", full_response, sources, is_fallback)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": list(sources),
        "is_fallback": is_fallback,
        "feedback": None
    })
    st.session_state.history_pairs.append({"q": prompt, "a": full_response})