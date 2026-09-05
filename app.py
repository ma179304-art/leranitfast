# ═══════════════════════════════════════════════════════════════════════════════
#  MEDCONSULT AI  v3.0
#  Elite Clinical Intelligence — Exam-Optimised + Consultant-Grade Management
#  Stack: Streamlit · Groq LLaMA 3.3 70B · Qdrant Cloud · Sentence-Transformers
# ═══════════════════════════════════════════════════════════════════════════════
import re, os, json, textwrap
from pathlib import Path
from datetime import date
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedConsult AI | Elite Clinical Intelligence Created by Dr Muhammad Asif",
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
# llama-3.3-70b-versatile was deprecated by Groq (announced 2026-06-17).
# Preference order — the first one actually available on the account wins.
GROQ_MODEL_CHAIN = [
    "openai/gpt-oss-120b",      # recommended replacement for llama-3.3-70b
    "qwen/qwen3.6-27b",         # multimodal alternative
    "moonshotai/kimi-k2-instruct-0905",
    "openai/gpt-oss-20b",       # smaller/faster fallback
]
GROQ_MODEL    = GROQ_MODEL_CHAIN[0]   # overridden at runtime by resolve_model()
RAG_TOP_K     = 10        # retrieve more chunks → richer context
MAX_TOKENS    = 4096      # longer, comprehensive answers
TEMPERATURE   = 0.20      # factual & consistent
HISTORY_TURNS = 6         # conversation pairs to include
HISTORY_ANSWER_CHARS = 2500  # truncate stored answers when re-feeding as context
SCORE_THRESH  = 0.30      # minimum Qdrant relevance score

# ── WEB SEARCH ────────────────────────────────────────────────────────────────
SEARCH_TOP_K      = 5     # web results to feed the model
SEARCH_SNIPPET    = 900   # max chars kept per result
SEARCH_TIMEOUT    = 12    # seconds

# ── SKILL ─────────────────────────────────────────────────────────────────────
SKILL_PATH = Path(__file__).parent / "fcps2-general-surgery-mastery" / "SKILL.md"
SKILL_MAX_CHARS = 14000   # trim if the skill file grows

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
.qtype-fcps       { background:#fee2e2; color:#991b1b; border:1px solid #fca5a5; }
.qtype-web        { background:#e0f2fe; color:#075985; border:1px solid #7dd3fc; }

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
    "fcps2_surgery":    ("🇵🇰 FCPS-II SURGERY",   "qtype-fcps"),
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
#  SKILL: FCPS-II GENERAL SURGERY MASTERY
# ═══════════════════════════════════════════════════════════════════════════════
# Fallback used if SKILL.md is not shipped alongside app.py.
_FCPS_FALLBACK = """
# FCPS Part II General Surgery Mastery

MISSION: act as a high-performance FCPS-II General Surgery preparation system for
CPSP trainees in Pakistan — not a textbook summariser. Maximise probability of
passing while building safe surgical reasoning across five domains: theory/MCOs,
SAQs, clinical & short cases, TOACS/viva, operative decision-making.

SOURCE HIERARCHY
1. Current official CPSP material (prospectus, notifications, clinical guidelines).
   If exam rules conflict with older books, current CPSP wins. Never present an old
   format as current — tell the candidate to verify on the CPSP e-portal.
2. Bailey & Love (primary daily text); Schwartz / Sabiston selectively for depth;
   Farquharson for operative technique.
3. Exam-focused material (MCO banks, SAQ/TOACS books, recalls) — treat recalls as
   potentially imperfect; never convert an unverified recall into a CPSP fact.
4. Current evidence: society guidelines, systematic reviews, landmark trials.

NEVER invent marks, station counts, timings, pass thresholds, eligibility rules or
format changes.

ANSWER ARCHITECTURE (for a disease topic)
One-line definition · classification (only if useful) · etiology/risk factors ·
concise pathophysiology · clinical features (symptoms / signs / red flags) ·
investigations (first-line, confirmatory, staging, preoperative) · diagnosis ·
focused differentials · management (initial stabilisation → medical → indications
for intervention → definitive surgery → alternatives → special situations) ·
complications · prognosis/follow-up · 3–7 FCPS pearls · high-yield viva questions.

For surgical topics always separate: resuscitation, diagnosis, optimisation,
temporising treatment, definitive treatment, operative options, postoperative care,
complications, recurrence/prevention. Use algorithms for management-heavy topics.

MCQ/MCO RULES
Clinical vignette, 4–5 plausible options, one best answer, one concept per question,
no accidental clues, no "all of the above". After the candidate commits, give:
Correct answer · Why · Why the others are wrong · FCPS pearl · Trap.
Never reveal the answer before the candidate answers when quizzing. Mix recall,
interpretation, diagnosis, next-best-step, management, complication, anatomy,
pathology and operative-decision items. Escalate difficulty Level 1→5. Retest
concepts the candidate got wrong.

SPECIAL MODES
"Teach me X" → concept, mechanism, diagnosis, management algorithm, pearls, viva
questions, 5–10 MCOs.
"Quiz me" → withhold answers until the candidate responds.
"Rapid fire" → one question at a time, wait for the answer.
"TOACS mode" → one station at a time with realistic instructions.
"Viva mode" → act as examiner, progressively harder.
"Long case mode" → presentation → examination → investigations → differential →
management → viva.
"SAQ mode" → question first, then mark the answer.
"Last-minute revision" → only highest-yield facts, algorithms, errors, traps.
"Make me a plan" / "I have X days left" → immediately convert remaining time into a
weighted, calendar-style plan built backward from the exam (Foundation →
Consolidation → Exam conversion → Final revision). In the last 7–14 days: no new
textbooks; high-yield revision, error log, timed practice, TOACS recognition, viva
fluency, operative indications, emergency algorithms.

GUARDRAILS
If the candidate's answer is unsafe, say so plainly, explain why, give the safer
approach, and distinguish exam logic from real-world care. Do not diagnose a real
patient with unwarranted certainty; recognise instability and recommend escalation.

STYLE: direct, clinically precise, exam-focused, high-yield, structured, encouraging
without sentimentality. No encyclopedic padding, no motivational speeches, no
fabricated CPSP facts. Simple question → answer it directly first, then the pearl.

WHAT TO DO NEXT (priority): correct unsafe knowledge → correct repeated errors →
master high-yield algorithms → complete missing core topics → increase question
volume → improve timed performance → convert to SAQ/TOACS/viva → memorise volatile
facts. Do not recommend more reading when retrieval is the real bottleneck.

END A SUBSTANTIAL SESSION WITH: what you mastered · what remains weak · next
retrieval task · one FCPS pearl.
"""

@st.cache_resource(show_spinner=False)
def load_skill() -> str:
    """Load the FCPS-II skill from SKILL.md, falling back to the embedded copy."""
    try:
        raw = SKILL_PATH.read_text(encoding="utf-8")
        raw = re.sub(r"(?s)^---.*?---\s*", "", raw, count=1)   # strip frontmatter
        return raw[:SKILL_MAX_CHARS]
    except Exception:
        return _FCPS_FALLBACK

FCPS_MODES = {
    "Auto":                "",
    "Teach me":            "Use 'Teach me' mode: concept, mechanism, diagnosis, management algorithm, FCPS pearls, viva questions, then 5–10 MCOs.",
    "Quiz me (MCOs)":      "Use 'Quiz me' mode. Ask exam-grade MCOs. Do NOT reveal any answer until the candidate replies.",
    "Rapid fire":          "Use 'Rapid fire' mode. One question at a time. Wait for the answer before continuing.",
    "TOACS station":       "Use 'TOACS mode'. Generate ONE realistic station at a time with candidate instructions and timing.",
    "Viva mode":           "Use 'Viva mode'. Act as the examiner. Ask progressively harder questions, one at a time.",
    "Long case":           "Use 'Long case mode'. Simulate presentation → examination → investigations → differential → management → viva.",
    "SAQ mode":            "Use 'SAQ mode'. Give the question first; after the candidate answers, mark it with a breakdown.",
    "Last-minute revision":"Use 'Last-minute revision' mode. Only the highest-yield facts, algorithms, traps and errors.",
    "Study plan":          "Use 'Make me a plan' mode. Build a calendar-style plan backward from the exam date. Do not stall on missing details — assume, state the assumption, continue.",
}

FCPS_REFERENCE_RULE = """

REFERENCING REQUIREMENT
Support your answer with sources. Rules:
• Attach a source inline, in brackets, to every major claim block — classification,
  investigation choice, management step, drug dose, staging, threshold or cut-off.
  Example: "...neoadjuvant chemotherapy is indicated [Bailey & Love, Breast chapter;
  NCCN Breast Cancer guideline]."
• End the answer with a '## 📚 References' section listing every source used.
• Cite by NAME and, where relevant, the guideline year or trial name:
  – Textbooks: Bailey & Love, Schwartz, Sabiston, Farquharson — name the chapter/topic.
  – Guidelines: NICE, NCCN, ESMO, WSES, ATLS, SAGES, IDSA, Tokyo Guidelines, CPSP.
  – Landmark trials: cite by trial acronym (e.g. ACOSOG Z0011, CRASH-2, FOxTROT).
• Where guidance differs between sources or between international and Pakistani/CPSP
  practice, say so explicitly and cite both.

ACCURACY RULES — these override the requirement above:
• NEVER invent page numbers, DOIs, PMIDs, ISBNs or URLs. Do not give a page number at all
  unless it came from supplied context; textbook editions differ.
• Cite a specific edition ONLY if you are certain of it; otherwise name the book without
  an edition number.
• Do not attribute a claim to a guideline you are not confident actually says it. If you
  are unsure of the source, write the claim and mark it '[standard teaching — verify]'
  rather than guessing an attribution.
• If a recommendation may have changed recently, flag it: '⚠️ verify against current
  guideline'.
• Never cite a trial you cannot name correctly."""

FCPS_WEB_REFERENCE_RULE = """
• Web results are supplied and numbered. For anything drawn from them, cite the number
  inline as [1], [2] and reproduce the title and URL in the References section exactly as
  given. Never alter a URL or invent one that is not in the supplied results."""


FCPS_CLOSERS = {

"Teach me": """Produce a COMPLETE FCPS-II teaching block on this topic. Emit every one of these
markdown headers in this order — do not omit or merge any, do not summarise:

## Definition
One or two lines.

## Classification
Markdown table. Include histological and molecular/staging subtypes where they exist.

## Etiology & Risk Factors
Table split into non-modifiable vs modifiable.

## Pathophysiology
Concise — 4–8 lines or a numbered cascade.

## Clinical Features
Three labelled groups: Symptoms · Examination findings · ⚠️ Red flags.

## Investigations
State explicitly: first-line · confirmatory/most useful · staging or severity · preoperative workup.
Say which test is WRONG and why, where trainees commonly err.

## Differential Diagnosis
Focused table: differential vs discriminating feature.

## Management
Sub-headed: Initial stabilisation → Medical treatment → Indications for intervention →
Definitive surgical treatment (named operations) → Alternatives → Special situations
(pregnancy, elderly, recurrent, metastatic, resource-limited). Use an explicit
algorithm with arrows. Include drug names and doses. Table when comparing regimens.

## Complications
Early vs late. Include named nerve/vessel injuries for operative topics.

## Prognosis & Follow-up

## 🔑 FCPS Pearls
5–7 bullets — the points that separate a strong candidate from a borderline one.

## 🎯 Viva Questions
6–8 questions, each with a one-line expected answer direction.

## 📚 References
Every source used, listed by name.

Be dense and high-yield. No filler sentences, no patient-leaflet tone. This is
postgraduate exam preparation — a short paragraph is a failed answer.""",

"Quiz me (MCOs)": """Generate exam-grade clinical MCOs on this topic. Vignette stem, 4–5 plausible options,
one best answer. Number them. Do NOT reveal any answer, explanation or hint now — wait
for the candidate to commit. End by asking for their answers.""",

"Rapid fire": """Ask ONE question only. Wait for the answer. Do not answer it yourself and do not
continue to a second question.""",

"TOACS station": """Generate ONE realistic TOACS station: station type (interactive/static), candidate
instructions, the stem/image description, time allowed, and the tasks. Do not give the
model answer until the candidate responds.""",

"Viva mode": """Act as the FCPS examiner. Ask one question, wait, then escalate difficulty based on the
answer. Open with an entry-level question on this topic. Do not lecture.""",

"Long case": """Run a long case on this topic. Present the case, then ask the candidate for their
history-taking priorities before revealing more. Proceed stepwise — do not dump the
whole case, examination, investigations and management at once.""",

"SAQ mode": """Write a realistic FCPS-II SAQ on this topic with mark allocation shown per part.
Give the question ONLY. Do not answer it — you will mark the candidate's attempt.""",

"Last-minute revision": """High-yield revision only. Emit: a one-line definition · the classification table ·
the management algorithm in arrow form · drug doses · the 8–10 most examinable facts ·
the 5 classic traps candidates fall for. No pathophysiology prose, no background.""",

"Study plan": """Build a calendar-style plan working backward from the exam date. Emit a week-by-week
table (week · topics · question volume · clinical/TOACS drill · revision target).
If the exam date or available hours weren't given, assume a reasonable value, state the
assumption in one line, and continue — do not stall by asking questions.""",
}
FCPS_CLOSERS["Auto"] = FCPS_CLOSERS["Teach me"]

# Modes where the model must NOT reveal content yet — referencing would leak the answer.
_NO_REF_MODES = {"Quiz me (MCOs)", "Rapid fire", "TOACS station", "Viva mode",
                 "Long case", "SAQ mode"}

def build_closer(mode: str, refs_on: bool, has_web: bool) -> str:
    closer = FCPS_CLOSERS.get(mode, FCPS_CLOSERS["Teach me"])
    if refs_on and mode not in _NO_REF_MODES:
        closer += FCPS_REFERENCE_RULE
        if has_web:
            closer += FCPS_WEB_REFERENCE_RULE
    return closer


def candidate_profile() -> str:
    """Exam date, weak topics and covered topics — injected so the tutor calibrates."""
    bits = []
    days = days_to_exam()
    if days is not None:
        bits.append(f"Exam date: {st.session_state.exam_date} ({days} days remaining).")
    if st.session_state.weak_topics:
        bits.append("Known weak areas: " + ", ".join(st.session_state.weak_topics) + ".")
    if st.session_state.covered:
        recent = list(st.session_state.covered)[-12:]
        bits.append("Already covered this session: " + ", ".join(recent) + ".")
    if not bits:
        return ""
    return ("\n\nCANDIDATE PROFILE\n" + " ".join(bits) +
            "\nWeight your answer accordingly: prioritise weak areas, avoid re-teaching "
            "covered ground in depth, and scale scope to the time remaining. "
            "Do not stall to ask for details you already have here.")


def days_to_exam():
    d = st.session_state.get("exam_date")
    if not d:
        return None
    return (d - date.today()).days


def export_session() -> str:
    """Whole conversation as a revision-ready markdown file."""
    lines = [
        "# MedConsult AI — FCPS-II Revision Session",
        f"_Exported {date.today().isoformat()} · model: {st.session_state.get('model','')}_",
    ]
    days = days_to_exam()
    if days is not None:
        lines.append(f"_Exam in {days} days_")
    if st.session_state.weak_topics:
        lines.append("\n**Weak areas flagged:** " + ", ".join(st.session_state.weak_topics))
    lines.append("\n---\n")
    for i, h in enumerate(st.session_state.history, 1):
        lines += [f"## {i}. {h['q'].strip()[:200]}", "", h["a"], "\n---\n"]
    return "\n".join(lines)


def detect_fcps_mode(question: str, chosen: str) -> str:
    """In Auto, infer the intended mode from the phrasing of the question."""
    if chosen != "Auto":
        return chosen
    q = question.strip()
    if re.search(r"(?i)\b(quiz me|test me|ask me|mcqs?|mcos?)\b", q):        return "Quiz me (MCOs)"
    if re.search(r"(?i)\b(rapid.?fire)\b", q):                              return "Rapid fire"
    if re.search(r"(?i)\btoacs\b", q):                                      return "TOACS station"
    if re.search(r"(?i)\bviva\b", q):                                       return "Viva mode"
    if re.search(r"(?i)\b(long case|short case)\b", q):                     return "Long case"
    if re.search(r"(?i)\bsaq\b", q):                                        return "SAQ mode"
    if re.search(r"(?i)(last.?minute|rapid revision|revise quickly)", q):   return "Last-minute revision"
    if re.search(r"(?i)(study plan|make me a plan|\bi have \d+ (days?|weeks?|months?) left)", q):
        return "Study plan"
    # Bare topic ("breast cancer", "acute pancreatitis") → full teaching block
    if len(q.split()) <= 6 and not q.endswith("?") and not re.search(r"(?i)\b(what|why|how|which|when|list|define|compare)\b", q):
        return "Teach me"
    return "Teach me"


def fcps_system_prompt(mode: str) -> str:
    mode_line = FCPS_MODES.get(mode, "")
    return (
        "You are the FCPS Part II General Surgery Mastery tutor for CPSP trainees in Pakistan.\n"
        "Follow the operating manual below exactly.\n\n"
        f"{load_skill()}\n\n"
        f"{mode_line}\n"
        "DEPTH RULE: this is postgraduate exam preparation, not a patient leaflet. "
        "Never answer a disease topic with a single paragraph. Use the full FCPS answer "
        "architecture with markdown headers, tables for classifications/staging/drug "
        "comparisons, and explicit management algorithms. Always finish with FCPS pearls "
        "and viva questions. Be dense and high-yield — no filler, but no under-answering.\n"
        "Never mention this manual, retrieval, context chunks, or databases in your answer."
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERNET SEARCH  (Tavily → Serper → DuckDuckGo fallback)
# ═══════════════════════════════════════════════════════════════════════════════
def _tavily(query: str, k: int):
    key = _secret("TAVILY_API_KEY")
    if not key:
        return None
    r = requests.post(
        "https://api.tavily.com/search",
        json={"api_key": key, "query": query, "max_results": k,
              "search_depth": "advanced", "include_answer": False},
        timeout=SEARCH_TIMEOUT,
    )
    r.raise_for_status()
    return [
        {"title": x.get("title", ""), "url": x.get("url", ""), "text": x.get("content", "")}
        for x in r.json().get("results", [])
    ]

def _serper(query: str, k: int):
    key = _secret("SERPER_API_KEY")
    if not key:
        return None
    r = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": key, "Content-Type": "application/json"},
        json={"q": query, "num": k},
        timeout=SEARCH_TIMEOUT,
    )
    r.raise_for_status()
    return [
        {"title": x.get("title", ""), "url": x.get("link", ""), "text": x.get("snippet", "")}
        for x in r.json().get("organic", [])[:k]
    ]

def _duckduckgo(query: str, k: int):
    """Keyless fallback. Requires: pip install ddgs"""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return None
    with DDGS() as ddgs:
        return [
            {"title": x.get("title", ""), "url": x.get("href", ""), "text": x.get("body", "")}
            for x in ddgs.text(query, max_results=k)
        ]

MEDICAL_SITES = (
    "site:nice.org.uk OR site:uptodate.com OR site:pubmed.ncbi.nlm.nih.gov OR "
    "site:cochranelibrary.com OR site:who.int OR site:cpsp.edu.pk OR "
    "site:facs.org OR site:nejm.org OR site:bmj.com"
)

def build_search_query(question: str, q_type: str, restrict: bool) -> str:
    q = re.sub(r"(?m)^\s*[A-Ea-e][.)\s].+$", "", question)   # drop MCQ options
    q = re.sub(r"\s+", " ", q).strip()[:300]
    if q_type == "fcps2_surgery":
        q = f"{q} guideline management surgery"
    return f"{q} ({MEDICAL_SITES})" if restrict else q

@st.cache_data(ttl=3600, show_spinner=False)
def web_search(query: str, k: int = SEARCH_TOP_K):
    """Try providers in order; return (results, provider_name, error)."""
    for name, fn in (("Tavily", _tavily), ("Serper", _serper), ("DuckDuckGo", _duckduckgo)):
        try:
            res = fn(query, k)
        except Exception as e:
            continue
        if res:
            return res, name, None
    return [], None, "No search provider available (set TAVILY_API_KEY / SERPER_API_KEY, or pip install ddgs)."

def format_web_context(results: list) -> str:
    blocks = []
    for i, r in enumerate(results, 1):
        txt = re.sub(r"\s+", " ", r["text"]).strip()[:SEARCH_SNIPPET]
        blocks.append(f"[{i}] {r['title']} — {r['url']}\n{txt}")
    return "\n\n".join(blocks)


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
#  MODEL RESOLUTION — pick the first model in the chain the account can actually use
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600, show_spinner=False)
def available_models() -> list:
    """Live list of chat models on this Groq account."""
    try:
        ids = [m.id for m in groq_client.models.list().data]
    except Exception:
        return []
    skip = ("whisper", "tts", "guard", "orpheus", "embed")
    return sorted(i for i in ids if not any(s in i.lower() for s in skip))

def resolve_model() -> str:
    models = available_models()
    if not models:
        return GROQ_MODEL_CHAIN[0]          # API unreachable — try the default
    for candidate in GROQ_MODEL_CHAIN:
        if candidate in models:
            return candidate
    return models[0]                        # nothing from the chain — use anything usable


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

    try:
        # qdrant-client ≥1.10
        hits = qdrant_client.query_points(
            collection_name = COLLECTION,
            query           = vec,
            limit           = RAG_TOP_K,
            score_threshold = SCORE_THRESH,
        ).points
    except AttributeError:
        # legacy client
        hits = qdrant_client.search(
            collection_name = COLLECTION,
            query_vector    = vec,
            limit           = RAG_TOP_K,
            score_threshold = SCORE_THRESH,
        )
    except Exception:
        return ""

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
def clean_response(text: str, keep_refs: bool = False) -> str:
    """keep_refs=True when web search was used — don't strip real citations."""
    for pattern, repl in _CLEAN:
        if keep_refs and ("^sources?" in pattern or "^references?" in pattern):
            continue
        text = re.sub(pattern, repl, text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
#  AI ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
def ask_ai(question: str, history: list, q_type: str,
           web_context: str = "", fcps_mode: str = "Auto", refs_on: bool = True):
    """Retrieve context (books + optional web), select prompt, stream response."""
    context = get_context(question, q_type)

    active_mode = detect_fcps_mode(question, fcps_mode) if q_type == "fcps2_surgery" else fcps_mode

    if q_type == "fcps2_surgery":
        sys_prompt = fcps_system_prompt(active_mode) + candidate_profile()
    else:
        sys_prompt = SYSTEM_PROMPTS.get(q_type, SYSTEM_PROMPTS["general_clinical"])

    if web_context:
        sys_prompt += (
            "\n\nWEB RESULTS ARE PROVIDED. Use them for anything current — guidelines, "
            "drug approvals, exam-format or CPSP notices, recent evidence. Prefer them over "
            "your own recall when they conflict, and cite inline as [1], [2] matching the "
            "numbered results. If the results don't answer the question, say so and answer "
            "from clinical knowledge. Do not fabricate citations or URLs."
        )

    messages = [{"role": "system", "content": sys_prompt}]

    # Include recent conversation history. Answers are truncated — full FCPS blocks
    # run to 8k tokens each and would exhaust the context window within a few turns.
    for h in history[-HISTORY_TURNS:]:
        prior = h["a"]
        if len(prior) > HISTORY_ANSWER_CHARS:
            prior = prior[:HISTORY_ANSWER_CHARS] + "\n…[earlier answer truncated]"
        messages += [
            {"role": "user",      "content": h["q"][:1500]},
            {"role": "assistant", "content": prior},
        ]

    ctx_block = f"\n\nREFERENCE CONTEXT:\n{context}" if context else ""
    web_block = f"\n\nWEB RESULTS:\n{web_context}" if web_context else ""

    if q_type == "fcps2_surgery":
        closing = build_closer(active_mode, refs_on, bool(web_context))
    else:
        closing = (
            "Answer exactly what was asked — directly and concisely. "
            "Do not add unrequested sections. Match the depth to the question."
        )
        if refs_on:
            closing += FCPS_REFERENCE_RULE
            if web_context:
                closing += FCPS_WEB_REFERENCE_RULE

    messages.append({
        "role": "user",
        "content": f"QUESTION:\n{question}{ctx_block}{web_block}\n\n{closing}",
    })

    return groq_client.chat.completions.create(
        model       = st.session_state.get("model", GROQ_MODEL),
        messages    = messages,
        temperature = TEMPERATURE,
        max_tokens  = 8192 if q_type == "fcps2_surgery" else MAX_TOKENS,
        stream      = True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
if "messages" not in st.session_state: st.session_state.messages = []
if "history"  not in st.session_state: st.session_state.history  = []
if "prefill"  not in st.session_state: st.session_state.prefill  = None
if "web_on"   not in st.session_state: st.session_state.web_on   = False
if "web_med"  not in st.session_state: st.session_state.web_med  = True
if "fcps_on"  not in st.session_state: st.session_state.fcps_on  = False
if "fcps_mode" not in st.session_state: st.session_state.fcps_mode = "Auto"
if "model"    not in st.session_state: st.session_state.model    = resolve_model()
if "refs_on"  not in st.session_state: st.session_state.refs_on  = True
if "exam_date"   not in st.session_state: st.session_state.exam_date   = None
if "weak_topics" not in st.session_state: st.session_state.weak_topics = []
if "covered"     not in st.session_state: st.session_state.covered     = []
if "last_q"      not in st.session_state: st.session_state.last_q      = None


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🩺 MedConsult AI v3.0")
    st.markdown(f"**Questions answered:** {len(st.session_state.history)}")
    st.divider()

    # ── Model ─────────────────────────────────────────────────────────────────
    st.markdown("#### 🧠 Model")
    _models = available_models()
    if _models:
        _idx = _models.index(st.session_state.model) if st.session_state.model in _models else 0
        st.session_state.model = st.selectbox("Active model", _models, index=_idx)
    else:
        st.caption("⚠️ Couldn't reach the Groq model list — check GROQ_API_KEY.")
        st.session_state.model = st.text_input("Active model", value=st.session_state.model)

    st.divider()

    # ── Internet search ───────────────────────────────────────────────────────
    st.markdown("#### 🌐 Internet Search")
    st.session_state.web_on = st.toggle(
        "Search the web", value=st.session_state.web_on,
        help="Fetches live results and grounds the answer in them with [1][2] citations.",
    )
    if st.session_state.web_on:
        st.session_state.web_med = st.checkbox(
            "Restrict to medical sources", value=st.session_state.web_med,
            help="NICE, PubMed, Cochrane, WHO, CPSP, BMJ, NEJM, ACS.",
        )

    st.divider()

    # ── References ────────────────────────────────────────────────────────────
    st.markdown("#### 📚 References")
    st.session_state.refs_on = st.toggle(
        "Cite sources", value=st.session_state.refs_on,
        help="Inline citations plus a References section. Textbooks and guidelines are "
             "cited by name; URLs appear only when web search supplies them.",
    )
    if st.session_state.refs_on and not st.session_state.web_on:
        st.caption("💡 Turn on web search for verifiable guideline citations with links.")

    st.divider()

    # ── FCPS-II skill ─────────────────────────────────────────────────────────
    st.markdown("#### 🇵🇰 FCPS-II General Surgery")
    st.session_state.fcps_on = st.toggle(
        "FCPS-II tutor mode", value=st.session_state.fcps_on,
        help="Loads the FCPS Part II General Surgery Mastery skill (CPSP-aligned).",
    )
    if st.session_state.fcps_on:
        st.session_state.fcps_mode = st.selectbox(
            "Session mode", list(FCPS_MODES.keys()),
            index=list(FCPS_MODES.keys()).index(st.session_state.fcps_mode),
        )
        st.caption("✅ SKILL.md loaded" if SKILL_PATH.exists() else "⚠️ SKILL.md not found — using built-in copy")

        # ── Exam countdown ────────────────────────────────────────────────────
        use_date = st.checkbox("Set exam date", value=st.session_state.exam_date is not None)
        if use_date:
            st.session_state.exam_date = st.date_input(
                "FCPS-II exam", value=st.session_state.exam_date or date.today(),
                min_value=date.today(),
            )
            d = days_to_exam()
            if d is not None:
                phase = ("🔴 Final revision" if d <= 14 else
                         "🟠 Exam conversion" if d <= 45 else
                         "🟡 Consolidation"   if d <= 120 else "🟢 Foundation")
                st.metric("Days remaining", d, delta=phase, delta_color="off")
        else:
            st.session_state.exam_date = None

        # ── Weak-area log ─────────────────────────────────────────────────────
        with st.expander(f"🎯 Weak areas ({len(st.session_state.weak_topics)})"):
            new_weak = st.text_input("Add topic", key="weak_in",
                                     placeholder="e.g. pancreatitis severity scoring")
            c1, c2 = st.columns(2)
            if c1.button("Add", use_container_width=True) and new_weak.strip():
                t = new_weak.strip()
                if t not in st.session_state.weak_topics:
                    st.session_state.weak_topics.append(t)
                st.rerun()
            if c2.button("Clear", use_container_width=True):
                st.session_state.weak_topics = []
                st.rerun()
            for t in st.session_state.weak_topics:
                st.markdown(f"- {t}")

    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history  = []
        st.session_state.covered  = []
        st.rerun()

    if st.session_state.history:
        st.download_button(
            "⬇️ Export session (.md)",
            data      = export_session(),
            file_name = f"fcps-revision-{date.today().isoformat()}.md",
            mime      = "text/markdown",
            use_container_width=True,
        )
        if st.button("🔄 Regenerate last answer", use_container_width=True):
            last = st.session_state.history.pop()
            st.session_state.messages = st.session_state.messages[:-2]
            st.session_state.prefill  = last["q"]
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
        f"<small>🔋 {st.session_state.model} · Qdrant Vector DB · Sentence-Transformers</small>",
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

    if st.session_state.fcps_on:
        st.markdown("**Quick start**")
        qc = st.columns(4)
        _quick = [
            ("📅 Study plan",  "Make me a study plan for FCPS-II General Surgery."),
            ("🎯 Quiz me",     "Quiz me with 10 mixed FCPS-II General Surgery MCOs."),
            ("🏥 TOACS",       "Give me a TOACS station."),
            ("⚡ Last-minute", "Last-minute revision: surgical emergencies."),
        ]
        for col, (lbl, q) in zip(qc, _quick):
            if col.button(lbl, use_container_width=True):
                st.session_state.prefill = q
                st.rerun()


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
    q_type = "fcps2_surgery" if st.session_state.fcps_on else classify_question(prompt)
    label, badge_class = _QTYPE_LABELS[q_type]

    # Render user message
    render_message("user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ── Internet search ───────────────────────────────────────────────────────
    web_context, web_results = "", []
    if st.session_state.web_on:
        searching = st.empty()
        searching.markdown("""
        <div class="thinking">
            <div style="display:flex;gap:5px"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>
            <div>Searching the web…</div>
        </div>""", unsafe_allow_html=True)
        query = build_search_query(prompt, q_type, st.session_state.web_med)
        web_results, provider, err = web_search(query, SEARCH_TOP_K)
        # Retry unrestricted if the medical-site filter returned nothing
        if not web_results and st.session_state.web_med:
            web_results, provider, err = web_search(
                build_search_query(prompt, q_type, False), SEARCH_TOP_K)
        searching.empty()
        if web_results:
            web_context = format_web_context(web_results)
        elif err:
            st.warning(f"🌐 {err}")

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
        stream = ask_ai(
            prompt,
            st.session_state.history,
            q_type,
            web_context = web_context,
            fcps_mode   = st.session_state.fcps_mode,
            refs_on     = st.session_state.refs_on,
        )
    except Exception as e:
        thinking.empty()
        msg = str(e)
        if "model_not_found" in msg or "decommissioned" in msg or "does not exist" in msg:
            st.error(
                f"⚠️ Model `{st.session_state.model}` isn't available on this account. "
                "Pick another from the sidebar — Groq retires models periodically "
                "(see console.groq.com/docs/deprecations)."
            )
            available_models.clear()   # drop the cache so the list refreshes
        else:
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
        final = clean_response(
            full_response,
            keep_refs = bool(web_results) or st.session_state.refs_on,
        )
        stream_box.markdown(final)

        if web_results:
            with st.expander(f"🌐 {len(web_results)} web sources"):
                for i, r in enumerate(web_results, 1):
                    st.markdown(f"**[{i}]** [{r['title'] or r['url']}]({r['url']})")

    # Persist to session
    st.session_state.messages.append({
        "role":    "assistant",
        "content": final,
        "q_type":  q_type,
    })
    st.session_state.history.append({"q": prompt, "a": final})
    st.session_state.last_q = prompt

    # Track covered ground so the tutor doesn't re-teach it
    topic = re.sub(r"\s+", " ", prompt).strip()[:60]
    if topic and topic not in st.session_state.covered:
        st.session_state.covered.append(topic)

    # ── Follow-up actions ─────────────────────────────────────────────────────
    if st.session_state.fcps_on:
        f1, f2, f3, f4 = st.columns(4)
        follow = [
            (f1, "🎯 Quiz me on this", f"Quiz me with 10 MCOs on: {topic}"),
            (f2, "🗣️ Viva me",         f"Viva mode on: {topic}"),
            (f3, "📝 SAQ",             f"SAQ mode on: {topic}"),
            (f4, "⚡ Condense",        f"Last-minute revision version of: {topic}"),
        ]
        for col, lbl, q in follow:
            if col.button(lbl, use_container_width=True, key=f"f_{lbl}_{len(st.session_state.history)}"):
                st.session_state.prefill = q
                st.rerun()

        if st.button("🚩 Flag this as a weak area", key=f"w_{len(st.session_state.history)}"):
            if topic not in st.session_state.weak_topics:
                st.session_state.weak_topics.append(topic)
            st.rerun()
