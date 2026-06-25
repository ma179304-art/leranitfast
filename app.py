# ═══════════════════════════════════════════════════════════════════════════════
#  MEDCONSULT AI  v3.0
#  Elite Clinical Intelligence — Exam-Optimised + Consultant-Grade Management
#  Stack: Streamlit · Groq LLaMA 3.3 70B · Qdrant Cloud · Sentence-Transformers
# ═══════════════════════════════════════════════════════════════════════════════
import re, os
import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedConsult AI | Elite Clinical Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CREDENTIALS ───────────────────────────────────────────────────────────────
# Priority: st.secrets → environment variable → hardcoded fallback
def _secret(key: str, fallback: str = "") -> str:
    """Resolve a secret from Streamlit secrets, env var, or hardcoded fallback."""
    try:
        return st.secrets[key]          # Streamlit Cloud / secrets.toml
    except Exception:
        return os.getenv(key, fallback) # local env var or hardcoded default

COLLECTION    = "medical_books"
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL    = "llama-3.3-70b-versatile"
RAG_TOP_K     = 10        # retrieve more chunks → richer context
MAX_TOKENS    = 4096      # longer, comprehensive answers
TEMPERATURE   = 0.20      # factual & consistent
HISTORY_TURNS = 6         # conversation pairs to include
SCORE_THRESH  = 0.30      # minimum Qdrant relevance score

# ═══════════════════════════════════════════════════════════════════════════════
#  CSS — CLEAN PROFESSIONAL MEDICAL UI
# ═══════════════════════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg:      #f0f4f8;
  --surface: #ffffff;
  --border:  #dde3ec;
  --accent:  #2563eb;
  --txt:     #1e293b;
  --txt2:    #64748b;
  --radius:  14px;
  --shadow:  0 2px 16px rgba(0,0,0,.07);
}
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
  font-family: 'Inter', sans-serif !important;
  background: var(--bg) !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── HEADER ─────────────────────────────────────────────────── */
.med-header {
  background: linear-gradient(135deg, #0f2442 0%, #1d4ed8 100%);
  padding: 18px 32px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  box-shadow: 0 4px 24px rgba(29,78,216,.3);
}
.logo { font-size: 1.45rem; font-weight: 700; color: #fff; letter-spacing: -.02em; }
.logo span { color: #7dd3fc; }
.live-badge {
  background: rgba(255,255,255,.12);
  border: 1px solid rgba(255,255,255,.22);
  color: #e2e8f0;
  font-size: .76rem;
  font-weight: 600;
  padding: 5px 14px;
  border-radius: 999px;
  display: flex;
  align-items: center;
  gap: 7px;
  letter-spacing: .03em;
}
.pulse-dot {
  width: 8px; height: 8px;
  background: #4ade80;
  border-radius: 50%;
  animation: pulse 1.8s infinite;
}
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(.6)} }

/* ── CHAT WRAPPER ───────────────────────────────────────────── */
.chat-wrap { max-width: 900px; margin: 0 auto; padding: 24px 20px 140px; }

/* ── WELCOME CARD ───────────────────────────────────────────── */
.welcome-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 36px 32px;
  text-align: center;
  box-shadow: var(--shadow);
  margin: 28px 0;
}
.welcome-card h2 { font-size: 1.45rem; font-weight: 700; color: var(--txt); margin-bottom: 12px; }
.welcome-card p  { color: var(--txt2); line-height: 1.75; font-size: .97rem; }
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 22px; }
.chip {
  background: #eff6ff; border: 1px solid #bfdbfe;
  color: var(--accent); font-size: .8rem; font-weight: 600;
  padding: 6px 14px; border-radius: 999px;
}

/* ── QTYPE BADGE ────────────────────────────────────────────── */
.qtype-badge {
  display: inline-flex; align-items: center; gap: 5px;
  font-size: .7rem; font-weight: 700; letter-spacing: .06em;
  padding: 3px 10px; border-radius: 999px;
  text-transform: uppercase; margin-bottom: 8px;
}
.qtype-exam       { background:#fef9c3; color:#854d0e; border:1px solid #fde047; }
.qtype-case       { background:#dbeafe; color:#1d4ed8; border:1px solid #93c5fd; }
.qtype-management { background:#dcfce7; color:#166534; border:1px solid #86efac; }
.qtype-patho      { background:#fce7f3; color:#9d174d; border:1px solid #f9a8d4; }
.qtype-pharma     { background:#f3e8ff; color:#6d28d9; border:1px solid #d8b4fe; }
.qtype-interp     { background:#ffedd5; color:#9a3412; border:1px solid #fdba74; }
.qtype-anatomy    { background:#e0f2fe; color:#0369a1; border:1px solid #7dd3fc; }
.qtype-procedure  { background:#ecfdf5; color:#065f46; border:1px solid #6ee7b7; }
.qtype-general    { background:#f1f5f9; color:#475569; border:1px solid #cbd5e1; }

/* ── CHAT MESSAGES ──────────────────────────────────────────── */
[data-testid="stChatMessage"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 16px 20px !important;
  margin: 6px 0 !important;
  box-shadow: var(--shadow) !important;
}

/* ── MARKDOWN: TABLES ───────────────────────────────────────── */
.stMarkdown table { width: 100%; border-collapse: collapse; font-size: .9rem; margin: 12px 0; }
.stMarkdown th    { background: #1d4ed8; color: #fff; padding: 9px 13px; text-align: left; }
.stMarkdown td    { padding: 8px 13px; border: 1px solid var(--border); }
.stMarkdown tr:nth-child(even) td { background: #f8fafc; }

/* ── MARKDOWN: INLINE ───────────────────────────────────────── */
.stMarkdown code {
  background: #eff6ff; color: #1d4ed8;
  padding: 2px 6px; border-radius: 4px; font-size: .9em;
}
.stMarkdown blockquote {
  border-left: 4px solid var(--accent);
  padding: 4px 12px; color: var(--txt2); margin: 8px 0;
}
.stMarkdown h1, .stMarkdown h2 { color: var(--txt); margin: 16px 0 8px; }
.stMarkdown h3 { color: var(--accent); margin: 12px 0 6px; }

/* ── THINKING INDICATOR ─────────────────────────────────────── */
.thinking {
  display: flex; align-items: center; gap: 12px;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 14px 20px;
  color: var(--txt2); font-size: .9rem; max-width: 400px;
  box-shadow: var(--shadow); margin: 6px 0;
}
.dot { width:8px; height:8px; background:var(--accent); border-radius:50%; animation:bounce .8s infinite alternate; }
.dot:nth-child(2){ animation-delay:.2s }
.dot:nth-child(3){ animation-delay:.4s }
@keyframes bounce{ from{transform:translateY(0)} to{transform:translateY(-6px)} }

/* ── INPUT BAR ──────────────────────────────────────────────── */
[data-testid="stChatInput"] {
  background: #ffffff !important;
  border: 2px solid var(--border) !important;
  border-radius: 24px !important;
  padding: 8px !important;
  box-shadow: 0 4px 24px rgba(0,0,0,.09) !important;
  position: sticky; bottom: 12px;
  transition: all .25s ease !important;
}
[data-testid="stChatInput"]:focus-within {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 4px rgba(37,99,235,.12), 0 8px 30px rgba(37,99,235,.15) !important;
  transform: translateY(-1px);
}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea:focus {
  color: #000 !important;
  caret-color: #000 !important;
  -webkit-text-fill-color: #000 !important;
  font-size: 1rem !important;
  font-family: 'Inter', sans-serif !important;
  background: transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder {
  color: #94a3b8 !important;
  -webkit-text-fill-color: #94a3b8 !important;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  QUESTION-TYPE CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════
_Q_PATTERNS = {
    "exam_mcq": [
        r"(?i)\b(which|what is the (most|best|first|next)|most (likely|appropriate|common)|toc|doc|drug of choice|treatment of choice|correct answer)\b",
        r"(?i)\b(mcq|usmle|mrcp|mrcs|fcps|plab|amc|step [123]|high.yield|exam question)\b",
        r"(?i)^[A-E][.)]\s",
    ],
    "case_scenario": [
        r"(?i)\b\d{1,3}[- ]?(year|yr)[s]?[- ]?old\b",
        r"(?i)\b(presents? with|brought to|comes? (in|to)|referred with|admitted with|a&e|er|emergency)\b",
        r"(?i)\b(hx:|o/e:|vitals?:|bp:|hr:|rr:|spo2:|temp:|on examination|complain[ts]?)\b",
    ],
    "management": [
        r"(?i)\b(how (do|would|should) (you|i|we)|how (to|is it) manag|management of|manage|treatment( of)?|protocol for|algorithm for|approach to)\b",
        r"(?i)\b(step[s]? (in|of|for)|immediate (management|treatment)|first.line|second.line|empiric|definitive treatment)\b",
    ],
    "pathophysiology": [
        r"(?i)\b(pathophysiology|pathogenesis|mechanism of|why does|how does .+ cause|what causes|etiology|aetiology|underlying (cause|mechanism))\b",
    ],
    "pharmacology": [
        r"(?i)\b(drug[s]?|medication|antibiotic|dose|dosage|pharmacokinetics|pharmacodynamics|mechanism of action of|side effect|adverse effect|contraindication|drug interaction|moa of)\b",
    ],
    "interpretation": [
        r"(?i)\b(interpret|what (does|do) (this|these|the)|findings?|report|ecg|ekg|x.?ray|ct scan|mri|ultrasound|uss|abg|arterial blood gas|lab result[s]?)\b",
    ],
    "anatomy": [
        r"(?i)\b(anatomy|anatomical|nerve supply|blood supply|lymphatic drainage|boundaries|relations?|surgical triangle|spaces?|compartment[s]?|layers?)\b",
    ],
    "procedure": [
        r"(?i)\b(procedure|technique|how (to|is it done)|steps? (for|of)|how (do|would) you perform|operation|incision|surgical (steps?|technique|approach))\b",
    ],
}

_QTYPE_LABELS = {
    "exam_mcq":         ("🎯 EXAM / MCQ",        "qtype-exam"),
    "case_scenario":    ("🏥 CLINICAL CASE",      "qtype-case"),
    "management":       ("📋 MANAGEMENT",         "qtype-management"),
    "pathophysiology":  ("🧬 PATHOPHYSIOLOGY",    "qtype-patho"),
    "pharmacology":     ("💊 PHARMACOLOGY",       "qtype-pharma"),
    "interpretation":   ("📊 INTERPRETATION",     "qtype-interp"),
    "anatomy":          ("🗺️ ANATOMY",            "qtype-anatomy"),
    "procedure":        ("🔧 PROCEDURE",          "qtype-procedure"),
    "general_clinical": ("🩺 CLINICAL",           "qtype-general"),
}

def classify_question(q: str) -> str:
    """Score each question type and return the best match."""
    scores = {
        k: sum(bool(re.search(p, q)) for p in v)
        for k, v in _Q_PATTERNS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general_clinical"


def enrich_query(q: str, q_type: str) -> str:
    """Prepend domain keywords to improve RAG retrieval precision."""
    if q_type == "exam_mcq":
        # Strip MCQ option lines (A. B. C. …) so we search the clinical scenario
        core = re.sub(r"(?m)^\s*[A-Ea-e][.)\s].+$", "", q)
        core = re.sub(r"\s+", " ", core).strip()
        return core if len(core) > 20 else q
    prefixes = {
        "management":      "management treatment protocol ",
        "pathophysiology": "pathophysiology mechanism pathogenesis ",
        "pharmacology":    "drug mechanism dose pharmacology ",
        "anatomy":         "anatomy blood supply nerve supply relations ",
        "procedure":       "surgical technique steps procedure ",
    }
    return prefixes.get(q_type, "") + q


# ═══════════════════════════════════════════════════════════════════════════════
#  SYSTEM PROMPTS — ONE PER QUESTION TYPE
# ═══════════════════════════════════════════════════════════════════════════════
_BASE = """
CORE RULES:
• Answer EXACTLY what is asked — no more, no less.
• Lead with the direct answer in the first sentence, always.
• Use structure (headers, tables, sections) ONLY when the question is complex enough to need it.
• For simple/direct questions: answer in plain prose, 2–5 sentences. No headers needed.
• For complex cases or management questions: use relevant sections only — skip any section that doesn't apply.
• NEVER pad with irrelevant sections just to look thorough.
• NEVER mention sources, textbooks, context chunks, retrieval, or databases.
• Include drug doses, scoring systems, and differentials only when directly relevant to the question.
• Emoji: ⚠️ warning · ✅ key action · 🔑 pearl · 💊 drug · 🔬 investigation · 🎯 direct answer
"""

SYSTEM_PROMPTS = {

# ── EXAM MCQ ──────────────────────────────────────────────────────────────────
"exam_mcq": _BASE + """
You are a world-class postgraduate exam coach (USMLE, MRCP, MRCS, FCPS, PLAB, AMC).

Always start with the correct answer immediately — bold and clear.
Then explain why it's correct in 2–3 sentences.
Then briefly state why each wrong option is incorrect.
Add high-yield pearls and the common trap only if genuinely useful.
Keep it tight. Exam candidates need clarity, not length.
""",

# ── CLINICAL CASE ─────────────────────────────────────────────────────────────
"case_scenario": _BASE + """
You are a world-class senior consultant teaching at a case conference.

State your working diagnosis immediately with brief reasoning.
Then cover only the sections the case actually needs — investigations, management, complications, pearls.
If the question only asks for a diagnosis, give the diagnosis and reasoning. Don't add management unprompted.
If the question asks for full workup, be comprehensive.
Match the depth of your answer to what was actually asked.
""",

# ── MANAGEMENT ────────────────────────────────────────────────────────────────
"management": _BASE + """
You are a world-class senior consultant providing management guidance.

Give immediate steps first, then definitive management.
Include drug doses when prescribing. Use a table only if there are multiple drugs.
Cover complications and monitoring only if relevant to what was asked.
Be practical and clinical — what would you actually do for this patient right now.
""",

# ── PATHOPHYSIOLOGY ───────────────────────────────────────────────────────────
"pathophysiology": _BASE + """
You are a world-class pathophysiologist and clinical educator.

Explain the mechanism clearly and directly. Start with the core process in plain language.
Use a step-by-step sequence only if the cascade is complex enough to need it.
Link mechanism to clinical features and treatment only if it adds value to the answer.
""",

# ── PHARMACOLOGY ──────────────────────────────────────────────────────────────
"pharmacology": _BASE + """
You are a world-class clinical pharmacologist.

If asked about a drug: state its class and mechanism first, then cover what was specifically asked
(dose / side effects / interactions / contraindications). Don't list everything if only one thing was asked.
Include a drug table only when comparing multiple agents.
""",

# ── INTERPRETATION ────────────────────────────────────────────────────────────
"interpretation": _BASE + """
You are a world-class diagnostician.

State what the findings show immediately. Explain each abnormal value clearly and concisely.
Give the most likely diagnosis and key differentials. Suggest next steps only if clearly needed.
Don't add sections that weren't asked about.
""",

# ── ANATOMY ───────────────────────────────────────────────────────────────────
"anatomy": _BASE + """
You are a world-class anatomist and surgical educator.

Answer the specific anatomical question asked. Cover blood supply, nerve supply, relations,
or clinical correlations only as relevant to what was asked. Be focused and precise.
""",

# ── PROCEDURE ─────────────────────────────────────────────────────────────────
"procedure": _BASE + """
You are a world-class surgical educator.

Give the steps clearly and in order. Include indications and key contraindications briefly.
Cover complications and pearls concisely. Don't pad with obvious or unrequested information.
""",

# ── GENERAL CLINICAL ──────────────────────────────────────────────────────────
"general_clinical": _BASE + """
You are a world-class senior consultant physician and surgical educator with 25+ years of experience.

Answer the question directly and concisely. Lead with the answer.
Use headers and structure only when the question genuinely requires it.
For a simple factual question — answer in prose, no headers needed.
For a complex clinical scenario — use relevant sections only (diagnosis, investigations, management, pearls).
Never include sections that weren't asked about. Quality over completeness.
""",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  RESOURCE LOADING
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_resources():
    # Read secrets HERE — st.secrets is fully ready at this point
    qdrant_url     = st.secrets.get("QDRANT_URL",     os.getenv("QDRANT_URL",     ""))
    qdrant_api_key = st.secrets.get("QDRANT_API_KEY", os.getenv("QDRANT_API_KEY", ""))
    groq_api_key   = st.secrets.get("GROQ_API_KEY",   os.getenv("GROQ_API_KEY",   ""))

    embed  = SentenceTransformer(EMBED_MODEL)
    qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    groq   = Groq(api_key=groq_api_key)
    return embed, qdrant, groq

embed_model, qdrant_client, groq_client = load_resources()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENHANCED RAG — RETRIEVE + DEDUPLICATE
# ═══════════════════════════════════════════════════════════════════════════════
def get_context(question: str, q_type: str) -> str:
    """
    Enrich the query, retrieve top_k chunks from Qdrant,
    deduplicate by leading-80-char fingerprint, and return formatted context.
    """
    enriched = enrich_query(question, q_type)
    vec      = embed_model.encode(enriched).tolist()

    hits = qdrant_client.search(
        collection_name = COLLECTION,
        query_vector    = vec,
        limit           = RAG_TOP_K,
        score_threshold = SCORE_THRESH,
    )

    seen, parts = set(), []
    for h in hits:
        txt = h.payload.get("text", "").strip()
        if not txt:
            continue
        fp = re.sub(r"\s+", " ", txt[:80]).lower()
        if fp in seen:
            continue
        seen.add(fp)
        parts.append(txt)

    return "\n\n---\n\n".join(parts) if parts else ""


# ═══════════════════════════════════════════════════════════════════════════════
#  RESPONSE CLEANER
# ═══════════════════════════════════════════════════════════════════════════════
_CLEAN = [
    (r"(?im)^sources?:.*$",                                        ""),
    (r"(?im)^references?:.*$",                                     ""),
    (r"(?i)\bbased on (the )?(context|reference|textbook|retrieval)\b[,.]?\s*", ""),
    (r"(?i)\baccording to (the )?(context|reference|textbook)\b[,.]?\s*",       ""),
    (r"(?i)\bthe (provided )?(context|reference[s]?)\b",           ""),
    (r"\n{3,}",                                                    "\n\n"),
]
def clean_response(text: str) -> str:
    for pattern, repl in _CLEAN:
        text = re.sub(pattern, repl, text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
#  AI ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
def ask_ai(question: str, history: list, q_type: str):
    """Retrieve context, select specialist prompt, stream response."""
    context    = get_context(question, q_type)
    sys_prompt = SYSTEM_PROMPTS.get(q_type, SYSTEM_PROMPTS["general_clinical"])

    messages = [{"role": "system", "content": sys_prompt}]

    # Include recent conversation history for multi-turn context
    for h in history[-HISTORY_TURNS:]:
        messages += [
            {"role": "user",      "content": h["q"]},
            {"role": "assistant", "content": h["a"]},
        ]

    ctx_block = f"\n\nREFERENCE CONTEXT:\n{context}" if context else ""
    messages.append({
        "role": "user",
        "content": (
            f"QUESTION:\n{question}{ctx_block}\n\n"
            "Answer exactly what was asked — directly and concisely. "
            "Do not add unrequested sections. Match the depth to the question."
        ),
    })

    return groq_client.chat.completions.create(
        model       = GROQ_MODEL,
        messages    = messages,
        temperature = TEMPERATURE,
        max_tokens  = MAX_TOKENS,
        stream      = True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
if "messages" not in st.session_state: st.session_state.messages = []
if "history"  not in st.session_state: st.session_state.history  = []
if "prefill"  not in st.session_state: st.session_state.prefill  = None


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🩺 MedConsult AI v3.0")
    st.markdown(f"**Questions answered:** {len(st.session_state.history)}")
    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history  = []
        st.rerun()
    st.divider()
    st.markdown("""
**Auto-detected question types:**
- 🎯 Exam MCQ
- 🏥 Clinical Case
- 📋 Management Protocol
- 🧬 Pathophysiology
- 💊 Pharmacology
- 📊 Investigation Interpretation
- 🗺️ Anatomy
- 🔧 Procedure
- 🩺 General Clinical
""")
    st.divider()
    st.markdown(
        "<small>🔋 LLaMA 3.3 70B · Qdrant Vector DB · Sentence-Transformers</small>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="med-header">
    <div class="logo">🩺 Med<span>Consult</span> AI</div>
    <div class="live-badge">
        <div class="pulse-dot"></div>
        Elite Clinical Intelligence Active
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  CHAT AREA
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

# ── Welcome screen ────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <h2>🩺 Elite Clinical Intelligence</h2>
        <p>
            Ask any clinical question and receive comprehensive, consultant-grade answers
            optimised for <strong>postgraduate examinations</strong> and
            <strong>real-world clinical practice</strong>.<br><br>
            The AI automatically detects your question type and applies the
            optimal specialist response structure.
        </p>
        <div class="chip-row">
            <span class="chip">🎯 Exam MCQs</span>
            <span class="chip">🏥 Clinical Cases</span>
            <span class="chip">📋 Management Plans</span>
            <span class="chip">🧬 Pathophysiology</span>
            <span class="chip">💊 Drug Doses</span>
            <span class="chip">📊 Investigation Interpretation</span>
            <span class="chip">🗺️ Anatomy</span>
            <span class="chip">🔧 Procedures</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Render a message ──────────────────────────────────────────────────────────
def render_message(role: str, content: str, q_type: str = None) -> None:
    with st.chat_message(role):
        if role == "assistant" and q_type:
            label, css_class = _QTYPE_LABELS.get(q_type, ("🩺 CLINICAL", "qtype-general"))
            st.markdown(
                f'<span class="qtype-badge {css_class}">{label}</span>',
                unsafe_allow_html=True,
            )
        st.markdown(content)


# ── Render conversation history ───────────────────────────────────────────────
for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"], msg.get("q_type"))

st.markdown("</div>", unsafe_allow_html=True)  # close .chat-wrap


# ═══════════════════════════════════════════════════════════════════════════════
#  CHAT INPUT
# ═══════════════════════════════════════════════════════════════════════════════
prompt = st.chat_input(
    "Ask anything clinical — exam MCQs, cases, management, pharmacology, anatomy, interpretation…"
)
if st.session_state.prefill:
    prompt = st.session_state.prefill
    st.session_state.prefill = None


# ═══════════════════════════════════════════════════════════════════════════════
#  HANDLE NEW QUESTION
# ═══════════════════════════════════════════════════════════════════════════════
if prompt:
    q_type = classify_question(prompt)
    label, badge_class = _QTYPE_LABELS[q_type]

    # Render user message
    render_message("user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Thinking indicator
    thinking = st.empty()
    thinking.markdown(f"""
    <div class="thinking">
        <div style="display:flex;gap:5px">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        </div>
        <div>Generating {label} response…</div>
    </div>""", unsafe_allow_html=True)

    # Stream response
    try:
        stream = ask_ai(prompt, st.session_state.history, q_type)
    except Exception as e:
        thinking.empty()
        st.error(f"⚠️ API error: {e}")
        st.stop()

    thinking.empty()
    full_response = ""

    with st.chat_message("assistant"):
        st.markdown(
            f'<span class="qtype-badge {badge_class}">{label}</span>',
            unsafe_allow_html=True,
        )
        stream_box = st.empty()
        for chunk in stream:
            token          = chunk.choices[0].delta.content or ""
            full_response += token
            stream_box.markdown(full_response + "▌")
        final = clean_response(full_response)
        stream_box.markdown(final)

    # Persist to session
    st.session_state.messages.append({
        "role":    "assistant",
        "content": final,
        "q_type":  q_type,
    })
    st.session_state.history.append({"q": prompt, "a": final})
