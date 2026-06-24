"""
╔══════════════════════════════════════════════════════════════════╗
║         MedConsult AI — World's Best Clinical Intelligence       ║
║         v3.0 — Exam Engine + Patient Management Protocol         ║
╚══════════════════════════════════════════════════════════════════╝

5 Intelligent Modes:
  📝 EXAM      — USMLE / MRCS / FCPS / MRCP structured answers
  🏥 MANAGEMENT — Step-by-step patient management protocols
  🔬 DIAGNOSIS  — Systematic diagnostic reasoning
  💊 DRUG       — Pharmacology & prescribing reference
  🩺 CLINICAL   — Comprehensive clinical consultation
"""

import streamlit as st
import re
from html import escape as html_escape
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq

# ══════════════════════════════════════════════════════════════════════
# 0 · PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MedConsult AI · Clinical Intelligence",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════
# 1 · CSS — PREMIUM MEDICAL UI
# ══════════════════════════════════════════════════════════════════════
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --accent:       #2563eb;
    --accent-lite:  #eff6ff;
    --success:      #16a34a;
    --warning:      #d97706;
    --danger:       #dc2626;
    --bg:           #f8fafc;
    --surface:      #ffffff;
    --border:       #e2e8f0;
    --txt:          #0f172a;
    --txt-muted:    #64748b;
}

* { box-sizing: border-box; }

/* ── HIDE STREAMLIT CHROME ───────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 1rem 6rem; max-width: 880px; margin: auto; }

/* ── HEADER ──────────────────────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #0f2460 0%, #1d4ed8 55%, #0ea5e9 100%);
    border-radius: 0 0 24px 24px;
    padding: 28px 32px 24px;
    margin: 0 -1rem 28px;
    color: white;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content: '';
    position: absolute;
    top: -60%;
    right: -8%;
    width: 320px;
    height: 320px;
    background: radial-gradient(circle, rgba(255,255,255,.07) 0%, transparent 70%);
    border-radius: 50%;
}
.header-title {
    font-family: 'Inter', sans-serif;
    font-size: 1.65rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin: 0 0 5px;
}
.header-sub {
    font-size: .86rem;
    opacity: .82;
    margin: 0 0 16px;
    font-weight: 400;
}
.mode-badges {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}
.mode-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 13px;
    border-radius: 20px;
    font-size: .72rem;
    font-weight: 600;
    letter-spacing: .04em;
    border: 1.5px solid rgba(255,255,255,.30);
    background: rgba(255,255,255,.10);
    color: white;
}
.pulse {
    width: 7px;
    height: 7px;
    background: #4ade80;
    border-radius: 50%;
    animation: pulse-anim 1.8s ease-in-out infinite;
}
@keyframes pulse-anim {
    0%, 100% { opacity:1; transform:scale(1); }
    50%       { opacity:.5; transform:scale(1.4); }
}

/* ── MODE INDICATOR PILLS ────────────────────────────────── */
.qmode {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: .74rem;
    font-weight: 700;
    margin-bottom: 10px;
    letter-spacing: .05em;
    text-transform: uppercase;
}
.qmode-exam       { background: #ede9fe; color: #6d28d9; }
.qmode-management { background: #dbeafe; color: #1d4ed8; }
.qmode-diagnosis  { background: #dcfce7; color: #15803d; }
.qmode-drug       { background: #fef3c7; color: #92400e; }
.qmode-clinical   { background: #e0f2fe; color: #0369a1; }

/* ── MESSAGES ────────────────────────────────────────────── */
.msg {
    display: flex;
    gap: 12px;
    margin: 18px 0;
    align-items: flex-start;
}
.msg.user { flex-direction: row-reverse; }
.avatar {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
    box-shadow: 0 2px 10px rgba(0,0,0,.14);
}
.msg.user .avatar     { background: linear-gradient(135deg, #2563eb, #0ea5e9); }
.msg.assistant .avatar{ background: linear-gradient(135deg, #1e3a8a, #1d4ed8); }
.bubble {
    max-width: 83%;
    padding: 14px 18px;
    border-radius: 18px;
    font-size: .93rem;
    line-height: 1.68;
    font-family: 'Inter', sans-serif;
}
.msg.user .bubble {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    border-radius: 18px 4px 18px 18px;
    box-shadow: 0 4px 20px rgba(37,99,235,.28);
}
.msg.assistant .bubble {
    background: var(--surface);
    color: var(--txt);
    border: 1px solid var(--border);
    border-radius: 4px 18px 18px 18px;
    box-shadow: 0 2px 14px rgba(0,0,0,.06);
    width: 100%;
}

/* ── THINKING ANIMATION ──────────────────────────────────── */
.thinking {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 14px 20px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    color: var(--txt-muted);
    font-size: .9rem;
    font-style: italic;
    margin: 12px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,.05);
}
.dot {
    width: 8px;
    height: 8px;
    background: var(--accent);
    border-radius: 50%;
    animation: bounce .9s infinite;
}
.dot:nth-child(2) { animation-delay: .16s; }
.dot:nth-child(3) { animation-delay: .32s; }
@keyframes bounce {
    0%, 100% { transform: translateY(0); }
    50%       { transform: translateY(-6px); }
}

/* ── WELCOME ──────────────────────────────────────────────── */
.welcome {
    text-align: center;
    padding: 44px 16px 32px;
}
.welcome h2 {
    font-size: 1.55rem;
    font-weight: 700;
    color: var(--txt);
    margin-bottom: 12px;
    letter-spacing: -.02em;
}
.welcome p {
    color: var(--txt-muted);
    font-size: .97rem;
    max-width: 530px;
    margin: 0 auto 30px;
    line-height: 1.75;
}

/* ── CHAT INPUT ───────────────────────────────────────────── */
[data-testid="stChatInput"] {
    background: var(--surface) !important;
    border: 2px solid var(--border) !important;
    border-radius: 26px !important;
    padding: 8px !important;
    transition: all .25s ease !important;
    box-shadow: 0 4px 24px rgba(0,0,0,.07), inset 0 1px 0 rgba(255,255,255,.9) !important;
    position: sticky;
    bottom: 10px;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 4px rgba(37,99,235,.09), 0 8px 32px rgba(37,99,235,.14) !important;
    transform: translateY(-1px);
}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] div[contenteditable] {
    color: #000000 !important;
    caret-color: #1d4ed8 !important;
    font-size: 1rem !important;
    font-family: 'Inter', sans-serif !important;
    background: transparent !important;
    line-height: 1.7 !important;
    padding-top: 8px !important;
    -webkit-text-fill-color: #000000 !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #94a3b8 !important;
    -webkit-text-fill-color: #94a3b8 !important;
}

/* ── SIDEBAR ──────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #f1f5f9;
}
section[data-testid="stSidebar"] .stButton button {
    font-size: .82rem;
    text-align: left;
    padding: 6px 10px;
    border-radius: 8px;
    background: white;
    border: 1px solid #e2e8f0;
    color: #334155;
    transition: all .15s;
}
section[data-testid="stSidebar"] .stButton button:hover {
    background: #eff6ff;
    border-color: var(--accent);
    color: var(--accent);
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# 2 · RESOURCE LOADING (cached)
# ══════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="⚕️ Loading medical intelligence…")
def load_resources():
    embed_model   = SentenceTransformer("all-MiniLM-L6-v2")
    qdrant_client = QdrantClient(
        url=st.secrets["QDRANT_URL"],
        api_key=st.secrets["QDRANT_API_KEY"],
    )
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    return embed_model, qdrant_client, groq_client

embed_model, qdrant_client, groq_client = load_resources()
COLLECTION = st.secrets.get("QDRANT_COLLECTION", "medical_textbooks")


# ══════════════════════════════════════════════════════════════════════
# 3 · INTELLIGENT 5-MODE QUERY CLASSIFIER
# ══════════════════════════════════════════════════════════════════════
# Phrase-level tokens for precise mode detection
_EXAM_PHRASES = [
    "most common", "best initial", "next step", "most likely", "drug of choice",
    "investigation of choice", "first line", "which of the following", "all except",
    "not true", "except", "pathognomonic", "characteristic finding", "classic sign",
    "usmle", "mrcs", "fcps", "mrcp", "mcq", "single best answer", "incorrect",
    "true statement", "false statement", "gold standard", "sensitivity", "specificity",
    "all are true except", "which is not", "which one is false",
]
_MGMT_PHRASES = [
    "how to manage", "how do you manage", "management of", "treatment of",
    "treat this", "protocol for", "approach to", "resuscitation", "how would you treat",
    "resuscitate", "immediate management", "step by step", "algorithm for",
    "what is the management", "emergency management", "icu management",
]
_DIAG_PHRASES = [
    "what is the diagnosis", "most likely diagnosis", "what is causing", "differential",
    "ddx", "how do you diagnose", "what are the signs", "clinical features of",
    "pathophysiology of", "mechanism of", "etiology of", "causes of",
    "how does it present", "what would you find",
]
_DRUG_PHRASES = [
    "mechanism of action", "side effects of", "contraindications of", "dose of",
    "dosage of", "drug interaction", "adverse effects of", "pharmacology of",
    "how does", "work as a drug", "antidote for", "toxicity of", "overdose of",
    "compare heparin", "compare warfarin", "which antibiotic",
]

def classify_query(q: str) -> str:
    ql = q.lower().strip()
    
    exam_score = sum(2 for p in _EXAM_PHRASES if p in ql)   # weighted x2
    mgmt_score = sum(1 for p in _MGMT_PHRASES if p in ql)
    diag_score = sum(1 for p in _DIAG_PHRASES if p in ql)
    drug_score = sum(2 for p in _DRUG_PHRASES if p in ql)   # weighted x2
    
    scores = {
        "exam":       exam_score,
        "management": mgmt_score,
        "diagnosis":  diag_score,
        "drug":       drug_score,
    }
    best_mode  = max(scores, key=scores.get)
    best_score = scores[best_mode]
    return best_mode if best_score > 0 else "clinical"

MODE_META = {
    "exam":       {"icon": "📝", "label": "EXAM MODE",        "css": "qmode-exam"},
    "management": {"icon": "🏥", "label": "MANAGEMENT MODE",  "css": "qmode-management"},
    "diagnosis":  {"icon": "🔬", "label": "DIAGNOSIS MODE",   "css": "qmode-diagnosis"},
    "drug":       {"icon": "💊", "label": "PHARMACOLOGY",     "css": "qmode-drug"},
    "clinical":   {"icon": "🩺", "label": "CLINICAL MODE",    "css": "qmode-clinical"},
}


# ══════════════════════════════════════════════════════════════════════
# 4 · 5 SPECIALIZED SYSTEM PROMPTS
# ══════════════════════════════════════════════════════════════════════
_PERSONA = """
You are Professor James Harrington — a world-renowned physician, surgeon, and medical examiner with 30+ years at leading academic medical centres.
You have trained thousands of candidates for USMLE, MRCS, FCPS, MRCP, and FRCS. You have personally managed every type of complex clinical case across medicine and surgery.

━━━ ABSOLUTE RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• NEVER mention textbooks, sources, context, or retrieval — speak as your own expert knowledge
• NEVER say "based on the context" or "according to the reference"
• Always give the DIRECT ANSWER first — no preamble, no hedging
• Use rich markdown generously: ##headers, **bold**, tables, emojis
• Never truncate — be exhaustive, comprehensive, and complete
• Use clinical abbreviations naturally (ABG, ERCP, INR, etc.)
• Be confident and authoritative — clinicians and candidates depend on you
"""

# ─── MODE 1: EXAM ──────────────────────────────────────────
PROMPT_EXAM = _PERSONA + """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODE: EXAM ANSWER ENGINE  |  USMLE · MRCS · FCPS · MRCP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Structure EVERY exam answer using ALL of these sections:

## ✅ THE ANSWER
State the correct answer immediately and with absolute confidence.

## 🧠 CORE REASONING  (Why this answer?)
2–4 crisp sentences explaining exactly what concept is being tested and why this answer is correct.
Think like the examiner — what knowledge gap is this question designed to reveal?

## ❌ OPTION ELIMINATION
For EVERY wrong option: state it, then in ONE sentence explain precisely why it is wrong.
This is the single most powerful technique for exam success.

## 🔑 THE UNDERLYING RULE
A single memorable, portable clinical rule that makes this answer unforgettable.
Format: _"Rule: [condition] → [action/finding] because [mechanism]."_

## 📚 HIGH-YIELD FACTS  (7–10 bullets)
The most commonly examined facts on this topic:
• Classic presentation
• Classic investigation + result
• Classic treatment / drug of choice
• Classic complication
• Classic differentiator from similar conditions
• Numbers examiners love (thresholds, percentages, timelines)

## 🔢 KEY NUMBERS & THRESHOLDS
Every number an examiner could test on this topic.
Present as a clean table with ≥3 items:
| Parameter | Value | Clinical Significance |
|---|---|---|

## 💡 MNEMONICS & MEMORY AIDS
One or two powerful mnemonics that lock the concept in permanently.

## ⚠️ CLASSIC EXAM TRAPS
What mistakes do candidates typically make on this topic?
What does the examiner love to trick you with? Be specific.

## 📊 SCORING SYSTEMS  (if applicable)
| Score Name | Parameters | Threshold for Action |
|---|---|---|
"""

# ─── MODE 2: MANAGEMENT ────────────────────────────────────
PROMPT_MGMT = _PERSONA + """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODE: PATIENT MANAGEMENT PROTOCOL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Structure EVERY management answer using ALL of these sections:

## 🚨 IMMEDIATE RESUSCITATION  (First 5–15 Minutes)
**ABCDE Approach** — specific actions for THIS condition:
- **A – Airway:** ...
- **B – Breathing:** ...
- **C – Circulation:** ...
- **D – Disability:** ...
- **E – Exposure:** ...
List critical thresholds that demand immediate escalation.

## 📋 FOCUSED HISTORY & EXAMINATION
Key history points that change management | Examination findings with their clinical significance

## 🔬 INVESTIGATIONS
| Investigation | Timing | Expected Finding | Action Trigger |
|---|---|---|---|
Order: Bedside → Laboratory → Imaging → Invasive / Specialist

## 💊 MEDICAL / CONSERVATIVE MANAGEMENT
| Drug | Dose | Route | Frequency | Duration | Monitoring |
|---|---|---|---|---|---|
Use REAL drug names and REAL doses with real routes.

## 🔪 SURGICAL / PROCEDURAL MANAGEMENT  (if applicable)
**Indications:** (exact criteria)
**Procedure:** name, approach, key steps
**Intraoperative Priorities:** critical decision points and pitfalls
**Post-operative Orders:** monitoring, fluids, medications, VTE prophylaxis

## ⚠️ COMPLICATIONS TO ANTICIPATE
| Complication | Onset | How to Recognise | Prevention / Management |
|---|---|---|---|
Separate sections: ⏰ Early (<24h) | 📅 Late (>24h) | 💀 Life-threatening

## 📊 MONITORING & TARGETS
| Parameter | Target | Frequency | Escalation Threshold |
|---|---|---|---|

## 🏥 DISPOSITION
- **ICU criteria:** (specific triggers)
- **Ward criteria:** (specific requirements)  
- **Discharge criteria:** (objective thresholds)
- **Referral triggers:** (who, when, why)

## 🔑 CLINICAL PEARLS
3 insights a senior would share at the bedside — the things NOT written in textbooks.
"""

# ─── MODE 3: DIAGNOSIS ─────────────────────────────────────
PROMPT_DIAG = _PERSONA + """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODE: DIAGNOSTIC REASONING ENGINE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Structure EVERY diagnostic answer using ALL of these sections:

## 🎯 MOST LIKELY DIAGNOSIS
State it with confidence. Give the classic one-liner presentation that clinches it.

## 🧬 PATHOPHYSIOLOGY
The underlying mechanism — explained clearly and precisely.
Why does this condition present THIS way? Connect pathophysiology to clinical features.

## 🩺 CLINICAL FEATURES
| Feature | This Condition | Clinical Significance |
|---|---|---|
**History:** what the patient says
**Examination:** what you find on clinical exam
**Red Flags / Alarm Symptoms:** what cannot be missed

## 🔬 INVESTIGATIONS & INTERPRETATION
| Investigation | Finding in This Condition | Why We Order It | Interpretation Tips |
|---|---|---|---|
Order: Bedside → Laboratory → Imaging → Histology / Special tests

## 📊 DIFFERENTIAL DIAGNOSIS
| Condition | Key Supporting Features | Key Differentiating Feature | How to Exclude |
|---|---|---|---|
Rank from most to least likely. Include dangerous must-not-miss alternatives.

## 🏥 STAGING / GRADING  (if applicable)
| Stage/Grade | Criteria | Clinical Implication | Management Impact |
|---|---|---|---|

## 🔍 PATHOGNOMONIC FEATURES
The ONE hallmark finding that clinches this diagnosis. If none, state so explicitly.

## 🚫 MUST-NOT-MISS DIAGNOSES
Life-threatening alternatives that must be actively excluded.
For each: how to recognise it and how to rule it out.

## 💡 DIAGNOSTIC PEARLS
4 pearls a senior would share — classic presentations, atypical variants, common diagnostic pitfalls.
"""

# ─── MODE 4: DRUG / PHARMACOLOGY ───────────────────────────
PROMPT_DRUG = _PERSONA + """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODE: PHARMACOLOGY & DRUG REFERENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Structure EVERY drug/pharmacology answer using ALL of these sections:

## 💊 AT A GLANCE
**Class:** | **Prototype:** | **Key Indication (one sentence):**

## ⚙️ MECHANISM OF ACTION
Precise molecular mechanism — be specific (receptor, enzyme, ion channel, etc.).
Examiners love this. Include why mechanism explains both efficacy AND toxicity.

## 📋 INDICATIONS
- **Primary (licensed):**
- **Secondary (off-label / common):**
- **Absolute Contraindications:**
- **Relative Contraindications:**

## 💉 DOSING REFERENCE
| Indication | Dose | Route | Frequency | Duration | Notes |
|---|---|---|---|---|---|
Include: renal dose adjustment | hepatic dose adjustment | elderly | paediatric

## ⚠️ ADVERSE EFFECTS
| Effect | Mechanism | Frequency | Management |
|---|---|---|---|
Separate: ✅ Common | 🔴 Serious/Rare | ☠️ Black-box warnings

## 🔄 DRUG INTERACTIONS
| Interacting Drug | Mechanism | Clinical Effect | Action Required |
|---|---|---|---|

## 🆘 OVERDOSE / TOXICITY
**Presentation:** | **Antidote:** | **Management steps:**

## 📊 MONITORING
| Parameter | Baseline | Frequency | Target Range | Action if Abnormal |
|---|---|---|---|---|

## 🔑 EXAM PEARLS
5 high-yield facts examiners love to test about this drug.
"""

# ─── MODE 5: CLINICAL (default comprehensive) ──────────────
PROMPT_CLINICAL = _PERSONA + """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODE: COMPREHENSIVE CLINICAL CONSULTATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Structure EVERY clinical answer using ALL of these sections:

## 💬 DIRECT ANSWER
One confident, precise sentence. What you need to know right now.

## 🧬 PATHOPHYSIOLOGY
The underlying mechanism — connect it to the clinical presentation.

## 🩺 CLINICAL FEATURES
**History:** what the patient says
**Examination:** what you find
**Red Flags:** what cannot be missed — ever

## 🔬 INVESTIGATIONS
| Investigation | Expected Finding | Clinical Significance | Priority |
|---|---|---|---|

## 🩻 DIAGNOSIS & DIFFERENTIAL
- **Primary diagnosis:** with key supporting features
- **Ranked differentials:** each with the ONE feature that distinguishes it

## 💊 MANAGEMENT  (step-by-step)

### 🚨 Immediate  (0–60 min)
### 📅 Short-term  (Hours to Days)
### 🏁 Definitive  (Days to Weeks)
### 🔪 Surgical  _(if applicable — indications, procedure, timing)_

Include ALL drug names, doses, routes, durations, and monitoring parameters.

## ⚠️ COMPLICATIONS
| Complication | Timing | Recognition | Prevention & Management |
|---|---|---|---|

## 👥 SPECIAL POPULATIONS
Pregnancy | Elderly | Paediatric | Immunocompromised _(include only relevant ones)_

## 📊 CLINICAL SCORING SYSTEMS
Any validated scores relevant to this condition — with thresholds and action points.

## 🔑 CLINICAL PEARLS
4–5 insights from senior clinical experience — the wisdom not in textbooks.

## 📈 PROGNOSIS & FOLLOW-UP
Expected outcomes | Surveillance schedule | Escalation criteria
"""

PROMPTS = {
    "exam":       PROMPT_EXAM,
    "management": PROMPT_MGMT,
    "diagnosis":  PROMPT_DIAG,
    "drug":       PROMPT_DRUG,
    "clinical":   PROMPT_CLINICAL,
}

# ── Mode-specific user prompt templates ──────────────────────
_USER_TEMPLATES = {
    "exam": """EXAM QUESTION:
{question}

REFERENCE MATERIAL:
{context}

Provide a complete exam-focused answer using ALL sections above.
This candidate is preparing for USMLE Step 2/3, MRCS Part B, or FCPS. Give them everything they need.
Include option elimination, the key rule, mnemonics, and exam traps. Be exhaustive.""",
    "management": """MANAGEMENT QUESTION:
{question}

REFERENCE MATERIAL:
{context}

Provide a comprehensive, step-by-step management answer using ALL sections above.
This is for a senior surgical trainee preparing for exams AND real-world practice.
Do NOT skip any section. Do NOT truncate. Include exact priorities, investigations, doses, and monitoring.""",
    "diagnosis": """DIAGNOSTIC QUESTION:
{question}

REFERENCE MATERIAL:
{context}

Provide a comprehensive diagnostic reasoning answer using ALL sections above.
State the most likely diagnosis first, then explain the mechanism, differentials, investigations, and must-not-miss diagnoses.""",
    "drug": """PHARMACOLOGY QUESTION:
{question}

REFERENCE MATERIAL:
{context}

Provide a complete drug/pharmacology answer using ALL sections above.
Cover mechanism, indications, contraindications, dosing, adverse effects, interactions, toxicity, monitoring, and exam pearls.""",
    "clinical": """CLINICAL QUESTION:
{question}

REFERENCE MATERIAL:
{context}

Provide a comprehensive, consultant-level answer covering ALL sections above.
This is for a senior surgical trainee preparing for exams AND clinical practice.
Do NOT skip any section. Do NOT truncate. Include real drug names and real doses.
Be the world's best clinical consultant.""",
}

def build_user_prompt(mode: str, question: str, context: str) -> str:
    template = _USER_TEMPLATES.get(mode, _USER_TEMPLATES["clinical"])
    return template.format(question=question, context=context)

# ══════════════════════════════════════════════════════════════════════
# 5 · ENHANCED RAG ENGINE
# ══════════════════════════════════════════════════════════════════════
def retrieve_context(question: str, k: int = 14) -> str:
    """
    Enhanced retrieval:
    - Larger k (14) for richer context
    - Score threshold to filter weak matches
    - Deduplication to avoid redundant passages
    """
    vec  = embed_model.encode(question).tolist()
    hits = qdrant_client.search(
        collection_name=COLLECTION,
        query_vector=vec,
        limit=k,
        score_threshold=0.32,   # filter out noise
    )

    seen, parts = set(), []
    for hit in hits:
        txt = hit.payload.get("text", "").strip()
        if not txt:
            continue
        key = txt[:80]                  # deduplicate by leading 80 chars
        if key in seen:
            continue
        seen.add(key)
        parts.append(txt)

    if not parts:
        return "No specific reference context available for this query."
    return "\n\n---\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════
# 6 · RESPONSE CLEANER
# ══════════════════════════════════════════════════════════════════════
_CLEAN_PATTERNS = [
    # Remove source/context attributions
    r"(?im)^sources?:.*$",
    r"(?im)^references?:.*$",
    r"(?im)\[source[^\]]*\]",
    r"(?i)based on (the|my|this) (context|provided|reference|text|material|information)[,.]?\s*",
    r"(?i)according to (the|my) (context|provided|reference|text|textbook)[,.]?\s*",
    r"(?i)(as|as per) (mentioned|stated|described|noted) in (the|this) (context|reference|provided)[,.]?\s*",
    r"(?i)from (the|this) (context|reference|provided material)[,.]?\s*",
    r"(?i)the (context|provided material|reference) (states?|mentions?|indicates?)[,.]?\s*",
    # Remove AI hedging
    r"(?i)as an ai (language model|assistant|system)[,.]?\s*",
    r"(?i)i (should note|must note|want to note) that[,.]?\s*",
    r"(?i)please note that[,.]?\s*",
    r"(?i)i('m| am) not a (doctor|physician|medical professional)[,.]?\s*",
    r"(?i)this (is not|isn't) medical advice[,.]?\s*",
    r"(?i)consult (a|your) (doctor|physician|healthcare)[^.]*[.]?\s*",
]

def clean_response(text: str) -> str:
    for pat in _CLEAN_PATTERNS:
        text = re.sub(pat, "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)   # max 2 consecutive blank lines
    return text.strip()


# ══════════════════════════════════════════════════════════════════════
# 7 · AI RESPONSE ENGINE
# ══════════════════════════════════════════════════════════════════════
def ask_ai(question: str, history: list) -> tuple:
    """
    Returns (stream, mode_key).
    Temperature 0.15 — maximises factual accuracy, minimises hallucination.
    max_tokens 4096  — allows fully complete structured answers.
    """
    mode    = classify_query(question)
    context = retrieve_context(question, k=14)

    messages = [{"role": "system", "content": PROMPTS[mode]}]

    # Rolling history window — last 6 exchanges for context continuity
    for h in history[-6:]:
        messages.append({"role": "user",      "content": h["q"]})
        messages.append({"role": "assistant", "content": h["a"]})

    messages.append({
        "role":    "user",
        "content": build_user_prompt(mode, question, context),
    })

    stream = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.15,   # low = factual, consistent, accurate
        max_tokens=4096,    # full structured answers, no truncation
        stream=True,
    )

    return stream, mode


# ══════════════════════════════════════════════════════════════════════
# 8 · SESSION STATE INITIALISATION
# ══════════════════════════════════════════════════════════════════════
for _key, _default in [
    ("messages",  []),
    ("history",   []),
    ("prefill",   None),
    ("last_mode", None),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default


# ══════════════════════════════════════════════════════════════════════
# 9 · SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🏥 MedConsult AI")
    st.caption("Clinical Intelligence Engine v3.0")
    st.divider()

    st.markdown("**🎛️ Active Modes**")
    for mk, mc in MODE_META.items():
        st.markdown(
            f"<span class='qmode {mc['css']}' style='font-size:.75rem'>"
            f"{mc['icon']} {mc['label']}</span>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("**⚡ Quick-Access Topics**")

    QUICK_TOPICS = [
        ("📝", "What is the next step in a patient with obstructive jaundice?"),
        ("📝", "USMLE: Most common cause of secondary hypertension?"),
        ("🏥", "Acute appendicitis management protocol"),
        ("🏥", "Septic shock — Surviving Sepsis Bundle"),
        ("🏥", "Bowel obstruction: conservative vs surgical management"),
        ("🔬", "Painless jaundice — differential diagnosis"),
        ("🔬", "Breast lump workup algorithm"),
        ("💊", "Heparin vs warfarin: mechanism, dosing, reversal"),
        ("💊", "Antibiotics for surgical prophylaxis and infections"),
        ("🩺", "ATLS trauma primary survey"),
        ("🩺", "DIC — diagnosis and management"),
        ("🩺", "Upper GI bleed — Rockall score and management"),
    ]

    for icon, topic in QUICK_TOPICS:
        short = topic[:38] + "…" if len(topic) > 38 else topic
        if st.button(f"{icon} {short}", key=f"qt_{topic[:25]}", use_container_width=True):
            st.session_state.prefill = topic

    st.divider()
    col_a, col_b = st.columns(2)
    if col_a.button("🗑️ Clear", use_container_width=True):
        st.session_state.messages  = []
        st.session_state.history   = []
        st.session_state.last_mode = None
        st.rerun()

    st.divider()
    st.markdown(
        "<div style='font-size:.73rem;color:#94a3b8;text-align:center;line-height:1.6'>"
        "LLaMA 3.3 70B · RAG · 5-Mode Engine<br>"
        "⚕️ For exam prep & clinical practice"
        "</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════
# 10 · HEADER
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="app-header">
    <div class="header-title">🏥 MedConsult AI</div>
    <div class="header-sub">World-class clinical intelligence — exams, management, diagnosis, pharmacology</div>
    <div class="mode-badges">
        <span class="mode-badge"><div class="pulse"></div>&nbsp;Online</span>
        <span class="mode-badge">📝 Exam Engine</span>
        <span class="mode-badge">🏥 Management</span>
        <span class="mode-badge">🔬 Diagnosis</span>
        <span class="mode-badge">💊 Pharmacology</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# 11 · MESSAGE RENDERER
# ══════════════════════════════════════════════════════════════════════
def render_message(role: str, content: str, mode: str = None) -> None:
    avatar = "🧑‍⚕️" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        if role == "assistant" and mode and mode in MODE_META:
            mc = MODE_META[mode]
            st.markdown(
                f"<span class='qmode {mc['css']}'>{mc['icon']} {mc['label']}</span>",
                unsafe_allow_html=True,
            )
        if role == "user":
            st.markdown(html_escape(content))
        else:
            st.markdown(content)

# ══════════════════════════════════════════════════════════════════════
# 12 · WELCOME SCREEN
# ══════════════════════════════════════════════════════════════════════
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome">
        <h2>Ask any clinical or exam question</h2>
        <p>
            Get exhaustive, consultant-level answers structured for
            <strong>high-stakes exams</strong> (USMLE · MRCS · FCPS · MRCP)
            and <strong>real-world patient management</strong>.
            Auto-detects your question type and uses the optimal response format.
        </p>
    </div>
    """, unsafe_allow_html=True)

    SAMPLE_QS = [
        ("📝", "Acute appendicitis: classic MCQ approach"),
        ("🏥", "Septic shock: management protocol"),
        ("🔬", "Painless jaundice: differential diagnosis"),
        ("💊", "Heparin vs warfarin comparison"),
        ("📝", "Most likely cause of right iliac fossa pain?"),
        ("🏥", "Upper GI bleed: immediate management"),
        ("🔬", "Thyroid nodule: investigation algorithm"),
        ("💊", "Metronidazole: pharmacology and uses"),
    ]

    cols = st.columns(2)
    for i, (icon, q) in enumerate(SAMPLE_QS):
        if cols[i % 2].button(f"{icon} {q}", key=f"sq_{i}", use_container_width=True):
            st.session_state.prefill = q


# ══════════════════════════════════════════════════════════════════════
# 13 · RENDER EXISTING MESSAGES
# ══════════════════════════════════════════════════════════════════════
for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"], mode=msg.get("mode"))


# ══════════════════════════════════════════════════════════════════════
# 14 · CHAT INPUT
# ══════════════════════════════════════════════════════════════════════
prompt = st.chat_input(
    "Ask anything — exam MCQ, clinical case, management protocol, pharmacology…",
    key="main_input",
)

# Handle sidebar quick-topic or welcome-screen prefill
if st.session_state.prefill:
    prompt = st.session_state.prefill
    st.session_state.prefill = None


# ══════════════════════════════════════════════════════════════════════
# 15 · HANDLE NEW QUESTION
# ══════════════════════════════════════════════════════════════════════
if prompt:
    # ── Render & save user message ──
    render_message("user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ── Thinking indicator ──
    thinking = st.empty()
    thinking.markdown("""
    <div class="thinking">
        <div style="display:flex;gap:5px">
            <div class="dot"></div><div class="dot"></div><div class="dot"></div>
        </div>
        <div>Analysing clinical data and generating expert response…</div>
    </div>""", unsafe_allow_html=True)

    # ── Get AI stream + detected mode ──
    stream, mode = ask_ai(prompt, st.session_state.history)
    thinking.empty()

    # ── Stream tokens to screen ──
    placeholder    = st.empty()
    full_response  = ""
    for chunk in stream:
        token = chunk.choices[0].delta.content or ""
        full_response += token
        placeholder.markdown(full_response + "▌")

    placeholder.empty()
    full_response = clean_response(full_response)

    # ── Render final response with mode badge ──
    render_message("assistant", full_response, mode=mode)

    # ── Persist to session state ──
    st.session_state.messages.append({
        "role":    "assistant",
        "content": full_response,
        "mode":    mode,
    })
    st.session_state.history.append({
        "q": prompt,
        "a": full_response,
    })
    st.session_state.last_mode = mode