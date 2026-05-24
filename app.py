import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq
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
    --text:       #e2e8f0;
    --text2:      #94a3b8;
    --text3:      #64748b;
    --user-bg:    #151f30;
    --ai-bg:      #0d1420;
    --red:        #f87171;
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
    padding: 40px 20px 20px;
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
    line-height: 1.6; margin-bottom: 24px;
}

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

/* ── CHAT INPUT ── */
[data-testid="stChatInput"] {
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 14px !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}

/* Buttons style overriding */
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

/* Inline suggestion chips layout */
.suggestion-container {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin-top: 10px;
}

/* Scrollbar */
::-webkit-scrollbar       { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0.5rem !important; max-width: 860px; }

/* Spinner */
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


# ── SESSION STATE INITIALIZATION ─────────────────────────────
if "messages"      not in st.session_state: st.session_state.messages      = []
if "history_pairs" not in st.session_state: st.session_state.history_pairs = []
if "input_query"   not in st.session_state: st.session_state.input_query   = ""


# ── RESOURCES ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models...")
def load_resources():
    embed  = SentenceTransformer("all-MiniLM-L6-v2")
    qdrant = QdrantClient(url=st.secrets["QDRANT_URL"], api_key=st.secrets["QDRANT_API_KEY"])
    groq   = Groq(api_key=st.secrets["GROQ_API_KEY"])
    return embed, qdrant, groq

# Wrap resource gathering in safety block to verify keys exist
try:
    embed_model, qdrant_client, groq_client = load_resources()
except Exception as e:
    st.error("🔑 Configuration Secret Error: Please verify that QDRANT_URL, QDRANT_API_KEY, and GROQ_API_KEY are configured in your secrets setup.")
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


# ── ASK FUNCTION ─────────────────────────────────────────────
def ask(question, history):
    q_vec = embed_model.encode([question]).tolist()[0]
    hits  = qdrant_client.search(
        collection_name="medical_books",
        query_vector=q_vec,
        limit=12,
        with_payload=True
    )

    context_parts, sources = [], set()
    for hit in hits:
        src = hit.payload.get("source", "Unknown")
        context_parts.append(f"[{src}]\n{hit.payload['text']}")
        sources.add(src.replace(".pdf", ""))
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

    stream = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.2,
        max_tokens=2000,
        stream=True
    )
    return stream, sources


# ── CONSTANTS / SAMPLE SUGGESTIONS ───────────────────────────
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


# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    books = get_book_list()

    st.markdown(f"""
    <div style="padding:16px 0 8px">
        <div style="font-size:1rem;font-weight:700;color:var(--text);margin-bottom:4px">🏥 Medical AI</div>
        <span class="badge-online"><span class="dot-pulse"></span>Online · {len(books)} books</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Updated to use stable st.rerun() call safely
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages      = []
        st.session_state.history_pairs = []
        st.session_state.input_query   = ""
        st.rerun()
        
    st.markdown(
        f'<div style="font-size:0.75rem;color:var(--text3);padding-top:8px;text-align:center">'
        f'{len(st.session_state.messages)//2} questions asked</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div style="color:var(--text3);font-size:0.7rem;margin-top:20px;line-height:1.6">'
        'For educational use only.<br>Not a substitute for clinical judgment.</div>',
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


# ── WELCOME SCREEN CHIP ENGINE ───────────────────────────────
def render_welcome():
    st.markdown("""
    <div class="welcome-wrap">
        <div class="welcome-icon">📖</div>
        <h2>What would you like to know?</h2>
        <p>Ask any clinical question or tap a high-frequency sample context below to populate the guidance prompt engine.</p>
    </div>
    """, unsafe_allow_html=True)
    
    cols = st.columns(2)
    for idx, (icon, text) in enumerate(SAMPLES):
        with cols[idx % 2]:
            # Updated interaction pattern setting query inside session state safely before execution
            if st.button(f"{icon} {text}", key=f"chip_{idx}", use_container_width=True):
                st.session_state.input_query = text
                st.rerun()


# ── RENDER MESSAGES ──────────────────────────────────────────
def render_message(role, content, sources=None):
    if role == "user":
        st.markdown(f"""
        <div class="msg-row user">
            <div class="avatar user-av">🧑‍⚕️</div>
            <div class="bubble user">{content}</div>
        </div>""", unsafe_allow_html=True)
    else:
        src_html = ""
        if sources:
            tags = "".join(f'<span class="src-tag">📖 {s}</span>' for s in sources)
            src_html = f'<div class="sources-bar"><span class="src-label">Sources:</span>{tags}</div>'
            
        st.markdown(f'<div class="msg-row"><div class="avatar ai-av">🤖</div><div class="bubble ai">', unsafe_allow_html=True)
        st.markdown(content)
        if src_html:
            st.markdown(src_html, unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)


# Show historical bubbles or initial welcome chip grid
if not st.session_state.messages:
    render_welcome()
else:
    for msg in st.session_state.messages:
        render_message(msg["role"], msg["content"], msg.get("sources"))


# ── INPUT ENGINE AND INTERACTION CAPTURE ─────────────────────
placeholder_text = "Ask a clinical question, e.g., 'Management of acute appendicitis'..."
chat_input_val = st.chat_input(placeholder=placeholder_text)

# Prioritize suggestion selection or keyboard layout forms safely
prompt = None
if st.session_state.input_query:
    prompt = st.session_state.input_query
    st.session_state.input_query = "" # Wipe state cleanly right after reading
elif chat_input_val:
    prompt = chat_input_val


# ── ENGINE PROCESSING & AMBIGUITY HANDLING ────────────────────
if prompt:
    # Render user query immediately inside view context
    render_message("user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Intent Classifier Validation Rules (Catches short/ambiguous phrases)
    stripped_prompt = prompt.strip().lower()
    if len(stripped_prompt.split()) <= 1 or stripped_prompt in ["appendicitis", "stomach", "cushing", "analgesia"]:
        # Ambiguity Fallback UI Component Integration
        clarification_text = f"""
        ### 🔍 Clarification Needed
        I detected your query regarding **"{prompt}"**, but it is broad. To provide an accurate, textbook-grounded answer, could you please specify your clinical objective?
        
        * Are you looking for the **diagnostic criteria and workflows**?
        * Are you inquiring about the **acute surgical intervention and management**?
        * Or are you exploring the **anatomy/pathophysiology** behind it?
        """
        render_message("assistant", clarification_text)
        st.session_state.messages.append({"role": "assistant", "content": clarification_text})
        
        # Microcopy Option Selection Blocks
        st.markdown('<div class="suggestion-container">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("📋 Diagnostic Criteria", key="clarify_diag", use_container_width=True):
                st.session_state.input_query = f"Diagnostic criteria and workup for {prompt}"
                st.rerun()
        with c2:
            if st.button("🔪 Surgical Management", key="clarify_surg", use_container_width=True):
                st.session_state.input_query = f"Management steps and treatment for acute {prompt}"
                st.rerun()
        with c3:
            if st.button("🧬 Pathophysiology", key="clarify_path", use_container_width=True):
                st.session_state.input_query = f"Pathophysiology and anatomy of {prompt}"
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        # High Confidence Execution Stream Flow
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("""
        <div class="thinking">
            <div class="thinking-dots">
                <span></span><span></span><span></span>
            </div>
            Searching textbook databases and synthesizing structured consultant response...
        </div>
        """, unsafe_allow_html=True)

        try:
            # Query backend vector database and Groq
            stream, sources = ask(prompt, st.session_state.history_pairs)
            thinking_placeholder.empty()

            # Stream buffer tokens to control progressive scrolling boundaries
            response_ph = st.empty()
            full_response = ""
            for chunk in stream:
                token = chunk.choices[0].delta.content or ""
                full_response += token
                response_ph.markdown(full_response + "▌")

            response_ph.empty()

            # Render structured text components beautifully
            render_message("assistant", full_response, sources)

            # Keep historical state tracks inside application layer cache
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": list(sources)
            })
            st.session_state.history_pairs.append({"q": prompt, "a": full_response})
            
        except Exception as e:
            thinking_placeholder.empty()
            error_msg = f"⚠️ **System Notice:** I encountered a latency or API configuration error while fetching answers. Please verify your credentials or try rephrasing your question."
            render_message("assistant", error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            
        st.rerun()