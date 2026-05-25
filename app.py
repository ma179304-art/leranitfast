import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq
import re
import time

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical AI Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────
# ADVANCED CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root{
    --bg:#060816;
    --panel:#0f172a;
    --panel2:#111c34;
    --border:#1e2d4d;
    --border2:#2d4371;
    --accent:#4f9eff;
    --accent2:#7c5cff;
    --accentGlow:rgba(79,158,255,.18);
    --txt:#edf2ff;
    --muted:#94a3b8;
    --muted2:#64748b;
    --green:#22c55e;
    --user:#16233c;
    --assistant:#0d1527;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"]{
    background: radial-gradient(circle at top, #0d1630 0%, #060816 50%);
    color: var(--txt);
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit */
#MainMenu, footer, header{
    visibility:hidden;
}

.block-container{
    padding-top:1rem!important;
    max-width:950px;
}

/* HEADER */
.main-header{
    display:flex;
    align-items:center;
    gap:18px;
    padding:18px 0 24px 0;
    margin-bottom:20px;
    border-bottom:1px solid var(--border);
}

.logo{
    width:60px;
    height:60px;
    border-radius:18px;
    display:flex;
    align-items:center;
    justify-content:center;
    background:linear-gradient(135deg,#4f9eff,#7c5cff);
    font-size:1.8rem;
    box-shadow:0 0 30px rgba(79,158,255,.35);
}

.main-header h1{
    margin:0;
    font-size:2rem;
    font-weight:800;
    letter-spacing:-1px;
}

.subtext{
    color:var(--muted);
    margin-top:4px;
    font-size:.95rem;
}

.online{
    display:inline-flex;
    align-items:center;
    gap:6px;
    background:rgba(34,197,94,.12);
    border:1px solid rgba(34,197,94,.25);
    color:#86efac;
    padding:4px 10px;
    border-radius:50px;
    margin-top:8px;
    font-size:.78rem;
}

.pulse{
    width:7px;
    height:7px;
    border-radius:50%;
    background:#22c55e;
    animation:pulse 1.5s infinite;
}

@keyframes pulse{
    0%{opacity:1}
    50%{opacity:.3}
    100%{opacity:1}
}

/* WELCOME */
.welcome{
    text-align:center;
    padding:80px 10px 50px;
}

.welcome h2{
    font-size:2rem;
    margin-bottom:10px;
}

.welcome p{
    color:var(--muted);
    max-width:650px;
    margin:auto;
    line-height:1.8;
    font-size:1.05rem;
}

/* SAMPLE CARDS */
.samples{
    display:grid;
    grid-template-columns:repeat(2,1fr);
    gap:14px;
    margin-top:35px;
}

.sample{
    background:rgba(17,28,52,.7);
    border:1px solid var(--border);
    padding:18px;
    border-radius:18px;
    transition:.2s;
    cursor:pointer;
}

.sample:hover{
    transform:translateY(-3px);
    border-color:var(--accent);
    box-shadow:0 0 25px rgba(79,158,255,.12);
}

/* CHAT */
.msg{
    display:flex;
    gap:14px;
    margin:22px 0;
    align-items:flex-start;
}

.msg.user{
    flex-direction:row-reverse;
}

.avatar{
    width:42px;
    height:42px;
    border-radius:14px;
    display:flex;
    align-items:center;
    justify-content:center;
    flex-shrink:0;
    font-size:1rem;
}

.user .avatar{
    background:linear-gradient(135deg,#23416f,#16233c);
    border:1px solid var(--border2);
}

.assistant .avatar{
    background:linear-gradient(135deg,#4f9eff,#7c5cff);
    box-shadow:0 0 20px rgba(79,158,255,.25);
}

.bubble{
    max-width:82%;
    padding:18px 22px;
    border-radius:22px;
    line-height:1.8;
    font-size:1rem;
}

.user .bubble{
    background:var(--user);
    border:1px solid var(--border2);
    border-top-right-radius:8px;
}

.assistant .bubble{
    background:var(--assistant);
    border:1px solid var(--border);
    border-top-left-radius:8px;
    border-left:3px solid var(--accent);
}

/* MARKDOWN */
.assistant .bubble h1,
.assistant .bubble h2,
.assistant .bubble h3{
    color:#8cbcff;
    margin-top:18px;
    margin-bottom:8px;
}

.assistant .bubble strong{
    color:white;
}

.assistant .bubble code{
    background:#17233f;
    padding:2px 8px;
    border-radius:6px;
}

/* BEST IMPROVED INPUT BAR */
[data-testid="stChatInput"]{
    background:linear-gradient(180deg,#0f172a,#111827)!important;
    border:2px solid #223354!important;
    border-radius:24px!important;
    padding:8px!important;
    transition:all .25s ease!important;
    box-shadow:
        0 10px 35px rgba(0,0,0,.35),
        inset 0 1px 0 rgba(255,255,255,.03);
    position:sticky;
    bottom:12px;
}

[data-testid="stChatInput"]:focus-within{
    border-color:var(--accent)!important;
    box-shadow:
        0 0 0 4px rgba(79,158,255,.12),
        0 12px 40px rgba(79,158,255,.18)!important;
    transform:translateY(-1px);
}

[data-testid="stChatInput"] textarea{
    color:var(--txt)!important;
    font-size:1.02rem!important;
    font-family:'Inter',sans-serif!important;
    background:transparent!important;
    line-height:1.7!important;
    padding-top:8px!important;
}

[data-testid="stChatInput"] textarea::placeholder{
    color:#7b8aa8!important;
    font-weight:500;
}

/* SEND BUTTON */
[data-testid="stChatInputSubmitButton"]{
    background:linear-gradient(135deg,#4f9eff,#7c5cff)!important;
    border-radius:14px!important;
    border:none!important;
    transition:.2s!important;
}

[data-testid="stChatInputSubmitButton"]:hover{
    transform:scale(1.05)!important;
    box-shadow:0 0 18px rgba(79,158,255,.35)!important;
}

/* THINKING */
.thinking{
    display:flex;
    align-items:center;
    gap:12px;
    color:var(--muted);
    padding:18px;
}

.dot{
    width:8px;
    height:8px;
    border-radius:50%;
    background:var(--accent);
    animation:bounce 1.2s infinite;
}

.dot:nth-child(2){animation-delay:.2s}
.dot:nth-child(3){animation-delay:.4s}

@keyframes bounce{
    0%,80%,100%{transform:translateY(0)}
    40%{transform:translateY(-8px)}
}

/* SIDEBAR */
[data-testid="stSidebar"]{
    background:#0d1425!important;
    border-right:1px solid var(--border)!important;
}

.stButton button{
    background:#111c34!important;
    color:#d7e5ff!important;
    border:1px solid var(--border)!important;
    border-radius:12px!important;
    transition:.2s!important;
}

.stButton button:hover{
    border-color:var(--accent)!important;
    transform:translateY(-1px);
}

/* SCROLL */
::-webkit-scrollbar{
    width:6px;
}

::-webkit-scrollbar-thumb{
    background:#334155;
    border-radius:20px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading medical AI...")
def load_resources():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    qdrant = QdrantClient(
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_API_KEY"]
    )

    groq_client = Groq(
        api_key=st.secrets["GROQ_API_KEY"]
    )

    return embed_model, qdrant, groq_client

embed_model, qdrant_client, groq_client = load_resources()

# ─────────────────────────────────────────────────────────────
# BETTER AI RESPONSE ENGINE
# ─────────────────────────────────────────────────────────────
def clean_response(text):
    text = re.sub(r"(?i)^sources:.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def ask_ai(question, history):

    q_vector = embed_model.encode([question]).tolist()[0]

    hits = qdrant_client.search(
        collection_name="medical_books",
        query_vector=q_vector,
        limit=14,
        with_payload=True
    )

    context_parts = []

    for hit in hits:
        txt = hit.payload.get("text", "")
        context_parts.append(txt)

    context = "\n\n".join(context_parts[:10])

    messages = [
        {
            "role":"system",
            "content":"""
You are an elite senior physician and medical educator.

Your responses must feel:
- intelligent
- clinically practical
- concise but high value
- confident
- deeply explanatory

IMPORTANT RULES:
- NEVER mention sources
- NEVER say "based on the textbook"
- NEVER say "according to context"
- NEVER mention retrieval
- Speak naturally like a brilliant consultant
- Give direct answers first
- Then explain reasoning
- Use markdown beautifully
- Make answers feel premium and human

If surgical/clinical:
- mention diagnosis
- investigations
- treatment
- complications
- pearls

Avoid robotic wording.
"""
        }
    ]

    for h in history[-5:]:
        messages.append({"role":"user","content":h["q"]})
        messages.append({"role":"assistant","content":h["a"]})

    messages.append({
        "role":"user",
        "content":f"""
QUESTION:
{question}

MEDICAL REFERENCE:
{context}

Generate a premium consultant-level answer.

Structure:
# Direct Answer
# Full Explanation
# Step by Step Management
# Clinical Pearl
"""
    })

    stream = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3,
        max_tokens=2200,
        stream=True
    )

    return stream

# ─────────────────────────────────────────────────────────────
# SESSION
# ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []

if "prefill" not in st.session_state:
    st.session_state.prefill = None

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
samples = [
    "Management of acute appendicitis",
    "Interpretation of elevated INR",
    "Complications of portal hypertension",
    "Neck abscess drainage steps",
    "Pre-operative cardiac risk assessment",
    "Acute pancreatitis severity scoring",
    "Appendicitis with normal WBC",
    "Post-operative fever causes"
]

with st.sidebar:

    st.markdown("## 🩺 Medical AI")

    st.markdown("""
    <div class="online">
        <div class="pulse"></div>
        AI Online
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Sample Questions")

    for s in samples:
        if st.button(s, use_container_width=True):
            st.session_state.prefill = s
            st.rerun()

    st.markdown("---")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="logo">🩺</div>
    <div>
        <h1>Medical AI Assistant</h1>
        <div class="subtext">
            Consultant-level medical reasoning powered by your AI textbook system
        </div>

        <div class="online">
            <div class="pulse"></div>
            Advanced Clinical Intelligence Active
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# WELCOME
# ─────────────────────────────────────────────────────────────
if not st.session_state.messages:

    st.markdown("""
    <div class="welcome">
        <h2>Ask any clinical question</h2>

        <p>
        Get intelligent consultant-style answers with deep explanations,
        diagnostic reasoning, management plans, and clinical pearls.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# RENDER CHAT
# ─────────────────────────────────────────────────────────────
def render_message(role, content):

    if role == "user":

        st.markdown(f"""
        <div class="msg user">
            <div class="avatar">🧑‍⚕️</div>
            <div class="bubble">
                {content}
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="msg assistant">
            <div class="avatar">🤖</div>
            <div class="bubble">
        """, unsafe_allow_html=True)

        st.markdown(content)

        st.markdown("""
            </div>
        </div>
        """, unsafe_allow_html=True)

# Render old messages
for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"])

# ─────────────────────────────────────────────────────────────
# IMPROVED ASK BAR
# ─────────────────────────────────────────────────────────────
prompt = st.chat_input(
    "Ask anything clinical... diagnosis, surgery, management, interpretation..."
)

if st.session_state.prefill:
    prompt = st.session_state.prefill
    st.session_state.prefill = None

# ─────────────────────────────────────────────────────────────
# HANDLE QUESTION
# ─────────────────────────────────────────────────────────────
if prompt:

    render_message("user", prompt)

    st.session_state.messages.append({
        "role":"user",
        "content":prompt
    })

    thinking = st.empty()

    thinking.markdown("""
    <div class="thinking">
        <div style="display:flex;gap:5px">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        </div>

        <div>
            Analyzing clinical data and generating expert response...
        </div>
    </div>
    """, unsafe_allow_html=True)

    stream = ask_ai(prompt, st.session_state.history)

    thinking.empty()

    response_placeholder = st.empty()

    full_response = ""

    for chunk in stream:

        token = chunk.choices[0].delta.content or ""

        full_response += token

        response_placeholder.markdown(full_response + "▌")

    response_placeholder.empty()

    full_response = clean_response(full_response)

    render_message("assistant", full_response)

    st.session_state.messages.append({
        "role":"assistant",
        "content":full_response
    })

    st.session_state.history.append({
        "q":prompt,
        "a":full_response
    })
