# ═══════════════════════════════════════════════════════════════════════════════
#  MEDCONSULT AI  v4.0  —  SAFETY-GOVERNED BUILD
#  Created by Dr Muhammad Asif (MBBS, FCPS General Surgery)
#
#  ⚠️  EVALUATION BUILD — UNDER ACTIVE TESTING. NOT FOR CLINICAL USE.
#
#  What changed from v3.0 (each fix maps to a documented failure):
#   1. Entity verification gate ....... blocks hallucinated drugs/guidelines
#   2. Citation binding ............... only retrieved chunks may be cited
#   3. Safety pre-pass ................ life-threat + layperson detection
#   4. Anti-sycophancy guard .......... blocks "confirm that X" agreement
#   5. Retrieval scoping .............. domain filter + demographic constraints
#   6. Router hardening ............... safety classification runs BEFORE q-type
#   7. Pipeline-emitted confidence .... model may no longer decorate with ✅
#
#  Stack: Streamlit · Groq · Qdrant Cloud · Sentence-Transformers
# ═══════════════════════════════════════════════════════════════════════════════
import re, os, json
from pathlib import Path
from datetime import date
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from groq import Groq

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MedConsult AI | Evaluation Build — Dr Muhammad Asif",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

AUTHOR      = "Dr Muhammad Asif"
AUTHOR_FULL = "Dr Muhammad Asif · MBBS, FCPS General Surgery"
BUILD       = "v4.0 · Evaluation Build"


# ── CREDENTIALS ───────────────────────────────────────────────────────────────
def _secret(key: str, fallback: str = "") -> str:
    """Resolve a secret from Streamlit secrets, then env var, then fallback."""
    try:
        return st.secrets[key]
    except Exception:
        return os.getenv(key, fallback)


COLLECTION   = "medical_books"
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

GROQ_MODEL_CHAIN = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
    "moonshotai/kimi-k2-instruct-0905",
    "openai/gpt-oss-20b",
]
GROQ_MODEL   = GROQ_MODEL_CHAIN[0]
AUX_MODEL_CHAIN = ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]  # cheap/fast guard calls

RAG_TOP_K    = 10
MAX_TOKENS   = 4096
TEMPERATURE  = 0.20
HISTORY_TURNS = 6
HISTORY_ANSWER_CHARS = 2500
SCORE_THRESH = 0.30

# ── SAFETY-LAYER CONSTANTS ────────────────────────────────────────────────────
CORPUS_YEAR       = 2025    # newest year the knowledge base can vouch for
ENTITY_THRESH     = 0.45    # Qdrant score below which a named entity is unverified
HIGH_CONF_SCORE   = 0.60    # top retrieval score needed for "grounded" confidence
GUARD_TIMEOUT     = 8

# ── WEB SEARCH ────────────────────────────────────────────────────────────────
SEARCH_TOP_K   = 5
SEARCH_SNIPPET = 900
SEARCH_TIMEOUT = 12

# ── SKILL ─────────────────────────────────────────────────────────────────────
SKILL_PATH = Path(__file__).parent / "fcps2-general-surgery-mastery" / "SKILL.md"
SKILL_MAX_CHARS = 14000


# ═══════════════════════════════════════════════════════════════════════════════
#  CSS
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

/* ── EVALUATION BANNER ──────────────────────────────────────── */
.eval-banner {
  background: #7f1d1d;
  color: #fff;
  font-size: .82rem;
  font-weight: 600;
  padding: 9px 32px;
  text-align: center;
  letter-spacing: .01em;
  border-bottom: 2px solid #dc2626;
}
.eval-banner strong { color: #fecaca; }

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
.byline { font-size: .74rem; color: #bfdbfe; font-weight: 500; margin-top: 2px; }
.live-badge {
  background: rgba(255,255,255,.12);
  border: 1px solid rgba(255,255,255,.22);
  color: #e2e8f0;
  font-size: .76rem; font-weight: 600;
  padding: 5px 14px; border-radius: 999px;
  display: flex; align-items: center; gap: 7px;
}
.pulse-dot { width:8px; height:8px; background:#fbbf24; border-radius:50%; animation:pulse 1.8s infinite; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(.6)} }

/* ── CHAT WRAPPER ───────────────────────────────────────────── */
.chat-wrap { max-width: 900px; margin: 0 auto; padding: 24px 20px 140px; }

/* ── WELCOME CARD ───────────────────────────────────────────── */
.welcome-card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 36px 32px;
  text-align: center; box-shadow: var(--shadow); margin: 28px 0;
}
.welcome-card h2 { font-size: 1.45rem; font-weight: 700; color: var(--txt); margin-bottom: 12px; }
.welcome-card p  { color: var(--txt2); line-height: 1.75; font-size: .97rem; }
.chip-row { display:flex; flex-wrap:wrap; gap:8px; justify-content:center; margin-top:22px; }
.chip { background:#eff6ff; border:1px solid #bfdbfe; color:var(--accent);
        font-size:.8rem; font-weight:600; padding:6px 14px; border-radius:999px; }

/* ── BADGES ─────────────────────────────────────────────────── */
.qtype-badge {
  display:inline-flex; align-items:center; gap:5px;
  font-size:.7rem; font-weight:700; letter-spacing:.06em;
  padding:3px 10px; border-radius:999px; text-transform:uppercase;
  margin-bottom:8px; margin-right:6px;
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
.qtype-emergency  { background:#fecaca; color:#7f1d1d; border:1px solid #f87171; }

/* confidence chips emitted by the pipeline, never by the model */
.conf-grounded   { background:#dcfce7; color:#166534; border:1px solid #86efac; }
.conf-partial    { background:#fef9c3; color:#854d0e; border:1px solid #fde047; }
.conf-unverified { background:#fee2e2; color:#991b1b; border:1px solid #fca5a5; }

/* ── SAFETY NOTICE ──────────────────────────────────────────── */
.safety-box {
  background:#fef2f2; border:1px solid #fecaca; border-left:4px solid #dc2626;
  border-radius:8px; padding:12px 16px; margin:10px 0;
  color:#7f1d1d; font-size:.9rem; line-height:1.6;
}
.gate-box {
  background:#fffbeb; border:1px solid #fde68a; border-left:4px solid #f59e0b;
  border-radius:8px; padding:12px 16px; margin:10px 0;
  color:#78350f; font-size:.9rem; line-height:1.6;
}

/* ── CHAT MESSAGES ──────────────────────────────────────────── */
[data-testid="stChatMessage"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 16px 20px !important; margin: 6px 0 !important;
  box-shadow: var(--shadow) !important;
}

/* ── MARKDOWN ───────────────────────────────────────────────── */
.stMarkdown table { width:100%; border-collapse:collapse; font-size:.9rem; margin:12px 0; }
.stMarkdown th    { background:#1d4ed8; color:#fff; padding:9px 13px; text-align:left; }
.stMarkdown td    { padding:8px 13px; border:1px solid var(--border); }
.stMarkdown tr:nth-child(even) td { background:#f8fafc; }
.stMarkdown code { background:#eff6ff; color:#1d4ed8; padding:2px 6px; border-radius:4px; font-size:.9em; }
.stMarkdown blockquote { border-left:4px solid var(--accent); padding:4px 12px; color:var(--txt2); margin:8px 0; }
.stMarkdown h1, .stMarkdown h2 { color:var(--txt); margin:16px 0 8px; }
.stMarkdown h3 { color:var(--accent); margin:12px 0 6px; }

/* ── THINKING ───────────────────────────────────────────────── */
.thinking {
  display:flex; align-items:center; gap:12px;
  background:var(--surface); border:1px solid var(--border);
  border-radius:var(--radius); padding:14px 20px;
  color:var(--txt2); font-size:.9rem; max-width:420px;
  box-shadow:var(--shadow); margin:6px 0;
}
.dot { width:8px; height:8px; background:var(--accent); border-radius:50%; animation:bounce .8s infinite alternate; }
.dot:nth-child(2){ animation-delay:.2s } .dot:nth-child(3){ animation-delay:.4s }
@keyframes bounce{ from{transform:translateY(0)} to{transform:translateY(-6px)} }

/* ── INPUT BAR ──────────────────────────────────────────────── */
[data-testid="stChatInput"] {
  background:#ffffff !important; border:2px solid var(--border) !important;
  border-radius:24px !important; padding:8px !important;
  box-shadow:0 4px 24px rgba(0,0,0,.09) !important;
  position:sticky; bottom:12px; transition:all .25s ease !important;
}
[data-testid="stChatInput"]:focus-within {
  border-color:var(--accent) !important;
  box-shadow:0 0 0 4px rgba(37,99,235,.12), 0 8px 30px rgba(37,99,235,.15) !important;
}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea:focus {
  color:#000 !important; caret-color:#000 !important;
  -webkit-text-fill-color:#000 !important;
  font-size:1rem !important; font-family:'Inter',sans-serif !important;
  background:transparent !important;
}
[data-testid="stChatInput"] textarea::placeholder {
  color:#94a3b8 !important; -webkit-text-fill-color:#94a3b8 !important;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  QUESTION-TYPE CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════
_Q_PATTERNS = {
    "exam_mcq": [
        r"(?i)\b(mcq|mco|usmle|mrcp|mrcs|fcps|plab|amc|step [123]|high.yield|exam question)\b",
        r"(?i)^[A-E][.)]\s",
        r"(?i)\b(correct answer|which of the following)\b",
    ],
    "case_scenario": [
        r"(?i)\b\d{1,3}[- ]?(year|yr)[s]?[- ]?old\b",
        r"(?i)\b(presents? with|brought to|comes? (in|to)|referred with|admitted with|a&e|er|emergency)\b",
        r"(?i)\b(hx:|o/e:|vitals?:|bp:|hr:|rr:|spo2:|temp:|on examination|post.?op|pod ?\d)\b",
    ],
    "management": [
        r"(?i)\b(how (do|would|should) (you|i|we)|how (to|is it) manag|management of|manage|protocol for|algorithm for|approach to)\b",
        r"(?i)\b(step[s]? (in|of|for)|immediate (management|treatment)|first.line|second.line|empiric|definitive treatment)\b",
    ],
    "pathophysiology": [
        r"(?i)\b(pathophysiology|pathogenesis|mechanism of|why does|how does .+ cause|what causes|etiology|aetiology)\b",
    ],
    "pharmacology": [
        r"(?i)\b(drug[s]?|medication|antibiotic|dose|dosage|pharmacokinetic|pharmacodynamic|mechanism of action of|side effect|adverse effect|contraindication|drug interaction|moa of)\b",
    ],
    "interpretation": [
        r"(?i)\b(interpret|what (does|do) (this|these|the)|findings?|ecg|ekg|x.?ray|ct scan|mri|ultrasound|uss|abg|arterial blood gas|lab result[s]?)\b",
    ],
    "anatomy": [
        r"(?i)\b(anatomy|anatomical|nerve supply|blood supply|lymphatic drainage|boundaries|relations?|surgical triangle|compartment[s]?)\b",
    ],
    "procedure": [
        r"(?i)\b(procedure|technique|steps? (for|of)|how (do|would) you perform|operation|incision|surgical (steps?|technique|approach))\b",
    ],
}

_QTYPE_LABELS = {
    "exam_mcq":         ("🎯 EXAM / MCQ",       "qtype-exam"),
    "case_scenario":    ("🏥 CLINICAL CASE",     "qtype-case"),
    "management":       ("📋 MANAGEMENT",        "qtype-management"),
    "pathophysiology":  ("🧬 PATHOPHYSIOLOGY",   "qtype-patho"),
    "pharmacology":     ("💊 PHARMACOLOGY",      "qtype-pharma"),
    "interpretation":   ("📊 INTERPRETATION",    "qtype-interp"),
    "anatomy":          ("🗺️ ANATOMY",           "qtype-anatomy"),
    "procedure":        ("🔧 PROCEDURE",         "qtype-procedure"),
    "general_clinical": ("🩺 CLINICAL",          "qtype-general"),
    "fcps2_surgery":    ("🇵🇰 FCPS-II SURGERY",  "qtype-fcps"),
    "emergency":        ("🚨 TIME-CRITICAL",     "qtype-emergency"),
}

_CONF_LABELS = {
    "grounded":   ("🟢 GROUNDED IN CORPUS",     "conf-grounded"),
    "partial":    ("🟡 PARTIALLY GROUNDED",     "conf-partial"),
    "unverified": ("🔴 UNVERIFIED — MODEL RECALL", "conf-unverified"),
}


def classify_question(q: str) -> str:
    scores = {k: sum(bool(re.search(p, q)) for p in v) for k, v in _Q_PATTERNS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general_clinical"


def enrich_query(q: str, q_type: str) -> str:
    if q_type == "exam_mcq":
        core = re.sub(r"(?m)^\s*[A-Ea-e][.)\s].+$", "", q)
        core = re.sub(r"\s+", " ", core).strip()
        return core if len(core) > 20 else q
    prefixes = {
        "management":      "management treatment protocol ",
        "pathophysiology": "pathophysiology mechanism pathogenesis ",
        "pharmacology":    "drug mechanism dose pharmacology ",
        "anatomy":         "anatomy blood supply nerve supply relations ",
        "procedure":       "surgical technique steps procedure ",
        "emergency":       "resuscitation emergency immediate management complication ",
    }
    return prefixes.get(q_type, "") + q


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 3a — SAFETY PRE-PASS  (runs BEFORE question-type routing)
#  Fixes: post-thyroidectomy stridor, POD4 AF, layperson prescribing
# ═══════════════════════════════════════════════════════════════════════════════
_LAYPERSON_PATTERNS = [
    r"(?i)\b(i'?m the patient|i am the patient|for myself|no doctor|can'?t see a doctor|"
    r"no hospital|what should i take|should i take|my leg|my chest|my stomach|i have a fever)\b",
]

_RED_FLAG_RULES = [
    # (name, [regex triggers], directive)
    ("AIRWAY — post-neck-surgery",
     [r"(?i)(thyroidectom|parathyroidectom|neck (surgery|dissection)|carotid endarterect)"],
     [r"(?i)\b(stridor|hoarse|voice change|neck swelling|difficulty breathing|"
      r"airway|dyspnoea|dyspnea|tight neck)\b"],
     "AIRWAY THREAT. Stridor, hoarseness or neck swelling after neck surgery means "
     "EXPANDING NECK HAEMATOMA or BILATERAL RECURRENT LARYNGEAL NERVE PALSY until "
     "excluded. The first action is bedside decompression: open the skin and strap "
     "sutures, evacuate the clot, call for senior/anaesthetic help, prepare for "
     "reintubation or surgical airway. Hypocalcaemia is a DIAGNOSIS OF EXCLUSION here "
     "and must NOT be the lead answer."),

    ("ANASTOMOTIC LEAK — occult",
     [r"(?i)(pod ?\d|post.?op(erative)? day ?\d|day ?\d+ (after|post)|anastomos|"
      r"resection|hemicolectom|colectom|anterior resection|gastrectom|oesophagectom|esophagectom)"],
     [r"(?i)\b(new.?onset af|atrial fibrillation|tachycard|hr ?1[0-9]{2}|oliguri|"
      r"urine output (down|low)|delirium|confusion|unwell|deteriorat|drain output|"
      r"turbid|feculent|hypotens)\b"],
     "OCCULT ANASTOMOTIC LEAK. Any unexplained deterioration on postoperative day 3–7 "
     "after a bowel anastomosis — new atrial fibrillation, tachycardia, oliguria, "
     "delirium, changed drain output — is an ANASTOMOTIC LEAK until excluded, even with "
     "a normal white cell count. Lead with this. Do NOT treat the arrhythmia or the "
     "catheter in isolation; beta-blockade in occult septic shock is harmful. Required "
     "actions: sepsis six, CT abdomen/pelvis with contrast, early senior surgical review, "
     "source control."),

    ("SHOCK / SEPSIS",
     [r"(?i)."],
     [r"(?i)\b(lactate ?[4-9]|lactate ?1[0-9]|bp ?[5-8][0-9]/|map ?<? ?65|"
      r"hypotens|septic shock|anuri|pH ?7\.[012])\b"],
     "HAEMODYNAMIC INSTABILITY. Address resuscitation and source control before any "
     "other question asked. State explicit endpoints and a reassessment interval."),

    ("NECROTISING SOFT-TISSUE INFECTION",
     [r"(?i)."],
     [r"(?i)(pain\W{0,6}(out of proportion|disproportionate)|out of proportion\W{0,6}to"
      r"|crepitus|dusky skin|skin necrosis|bullae|"
      r"(red|erythema|redness).{0,40}(spreading|advancing|ascending|moving up|going up))"],
     "POSSIBLE NECROTISING SOFT-TISSUE INFECTION. This is a surgical emergency. Do not "
     "delay for imaging or scoring systems. Lead with immediate surgical exploration."),

    ("PAEDIATRIC",
     [r"(?i)."],
     [r"(?i)\b(\d{1,2} ?kg|neonat|infant|newborn|\b[1-9] ?(month|year)s? old\b|paediatric|pediatric)\b"],
     "PAEDIATRIC PATIENT. Recompute EVERY dose per kilogram, state the adult ceiling "
     "dose that must not be exceeded, give maintenance fluid (4-2-1 rule) separately "
     "from any bolus, and give an explicit antibiotic STOP DATE. Never write "
     "'continue until source control'."),

    ("CAPACITY / CONSENT",
     [r"(?i)."],
     [r"(?i)(refus\w+ (a )?(stoma|surgery|operation|treatment)|declin\w+ (surgery|treatment)|"
      r"family insists|against (his|her|their) wishes|confused.{0,40}(consent|surgery))"],
     "CAPACITY QUESTION. Confusion is NOT incapacity. Capacity is decision-specific and "
     "must be formally assessed (understand / retain / weigh / communicate). A patient "
     "WITH capacity may refuse surgery and that refusal cannot be overridden by family. "
     "Address the capacity assessment and the less-invasive alternatives (e.g. "
     "self-expanding metal stent for obstructing left-sided colorectal cancer) BEFORE "
     "describing any operation."),
]


def detect_layperson(q: str) -> bool:
    return any(re.search(p, q) for p in _LAYPERSON_PATTERNS)


def safety_prepass(question: str) -> dict:
    """Deterministic red-flag detection. No LLM — must never fail open."""
    flags, directives = [], []
    for name, ctx_pats, trig_pats, directive in _RED_FLAG_RULES:
        ctx_hit  = any(re.search(p, question) for p in ctx_pats)
        trig_hit = any(re.search(p, question) for p in trig_pats)
        if ctx_hit and trig_hit:
            flags.append(name)
            directives.append(f"[{name}] {directive}")
    return {
        "flags": flags,
        "directives": directives,
        "life_threat": bool(flags),
        "layperson": detect_layperson(question),
    }


LAYPERSON_RESPONSE = """### This needs a clinician today, not a prescription

I can't give you antibiotic names or doses. Not because of a rule — because
choosing wrong here is dangerous, and what you're describing may not be simple
cellulitis.

**A red area spreading up a limb with fever needs assessment today.** If any of
these are present, go to an emergency department now rather than waiting:

- Pain that feels far worse than the skin looks
- The red edge advancing while you watch it, or marked spread in a few hours
- Blistering, dusky or blackened skin, or a crackling feeling under the skin
- Fever with shivering, confusion, vomiting, or feeling faint
- You have diabetes, a weakened immune system, or recent surgery or injury there

**Before you go:** draw around the edge of the redness with a pen and note the
time. That single mark tells the treating doctor how fast it is moving, and it
genuinely changes management.

Oral antibiotics started blindly can mask a deep infection that needs surgery.
Please get seen.
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 1 — ENTITY VERIFICATION GATE
#  Fixes: tenapanor-B (hallucinated drug), KDIGO 2026 (hallucinated guideline)
# ═══════════════════════════════════════════════════════════════════════════════
_DRUG_STEM = (r"(?:sentan|panor|mab|nib|tinib|ciclib|cillin|micin|mycin|azole|parin|xaban|"
              r"gliflozin|gliptin|sartan|pril|statin|prazole|floxacin|cycline|penem|dipine)")

# A drug stem carrying a hyphenated suffix — "tenapanor-B", "sotagliflozin-2".
# This shape is the classic hallucination bait and is graded HARD.
_DRUG_SUFFIXED = re.compile(rf"\b[A-Za-z]{{4,}}{_DRUG_STEM}-[A-Za-z0-9]{{1,3}}\b", re.I)
# A plain drug-like INN token — graded SOFT (may simply be outside a surgical corpus).
_DRUGLIKE      = re.compile(rf"\b[A-Za-z]{{4,}}{_DRUG_STEM}\b", re.I)

_BODIES = (r"KDIGO|NICE|NCCN|ESMO|WSES|ATLS|SAGES|IDSA|Tokyo Guidelines?|CPSP|EAST|ACS|"
           r"AHA|ESC|Surviving Sepsis|EAU|BTS|Cochrane")
# Bodies and years appear in either order in real prompts.
_BODY_THEN_YEAR = re.compile(rf"\b({_BODIES})\b[^.\n]{{0,40}}?\b(19|20)(\d{{2}})\b", re.I)
_YEAR_THEN_BODY = re.compile(rf"\b(19|20)(\d{{2}})\b[^.\n]{{0,25}}?\b({_BODIES})\b", re.I)

_TRIAL = re.compile(r"\b([A-Z]{3,}[A-Z0-9\-]{0,8})\b(?=\s*(trial|study|rct))")


def extract_entities(question: str) -> list:
    """
    Deterministic entity extraction. severity 'hard' → the model must refuse outright;
    'soft' → the model must say the entity is outside its knowledge base.
    """
    found = []

    for m in _DRUG_SUFFIXED.finditer(question):
        found.append({"name": m.group(0), "kind": "drug", "year": None, "severity": "hard"})
    for m in _DRUGLIKE.finditer(question):
        found.append({"name": m.group(0), "kind": "drug", "year": None, "severity": "soft"})

    for m in _BODY_THEN_YEAR.finditer(question):
        yr = int(m.group(2) + m.group(3))
        found.append({"name": f"{m.group(1)} {yr}", "kind": "guideline", "year": yr,
                      "severity": "soft"})
    for m in _YEAR_THEN_BODY.finditer(question):
        yr = int(m.group(1) + m.group(2))
        found.append({"name": f"{m.group(3)} {yr}", "kind": "guideline", "year": yr,
                      "severity": "soft"})

    for m in _TRIAL.finditer(question):
        found.append({"name": m.group(1), "kind": "trial", "year": None, "severity": "soft"})

    # De-duplicate. Drop any name that is a substring of a longer one already found,
    # so "tenapanor" never shadows "tenapanor-B".
    found.sort(key=lambda e: len(e["name"]), reverse=True)
    kept = []
    for e in found:
        low = e["name"].lower()
        if any(low in k["name"].lower() for k in kept):
            continue
        kept.append(e)
    return kept


def verify_entities(entities: list, chunk_texts: list) -> list:
    """Return the entities the knowledge base cannot vouch for, with severity."""
    unverified = []
    corpus_blob = " ".join(chunk_texts).lower()
    for e in entities:
        # A guideline dated after the corpus cutoff can never be verified — always hard.
        if e.get("year") and e["year"] > CORPUS_YEAR:
            unverified.append({**e, "severity": "hard",
                               "reason": f"dated {e['year']}, after the {CORPUS_YEAR} corpus cutoff"})
            continue
        if e["name"].lower() in corpus_blob:
            continue
        try:
            hits = _qdrant_search(embed_model.encode(e["name"]).tolist(), limit=3)
            top = max((float(getattr(h, "score", 0.0) or 0.0) for h in hits), default=0.0)
        except Exception:
            top = 0.0
        if top < ENTITY_THRESH:
            unverified.append({**e, "reason": "not found in the knowledge base"})
    return unverified


def build_entity_directive(unverified: list) -> str:
    if not unverified:
        return ""
    hard = [u for u in unverified if u.get("severity") == "hard"]
    soft = [u for u in unverified if u.get("severity") != "hard"]
    out = "\n\n=== ENTITY VERIFICATION GATE — HIGHEST PRIORITY ===\n"

    if hard:
        out += ("CANNOT BE VERIFIED TO EXIST: "
                + "; ".join(f"\"{u['name']}\" ({u['reason']})" for u in hard) + "\n"
                "For each of these you MUST:\n"
                "1. Open your answer by stating plainly that you cannot verify it exists.\n"
                "2. NOT describe its mechanism, class, dose, safety profile or interactions.\n"
                "3. NOT cite any source for it.\n"
                "4. NOT answer the question as asked.\n"
                "If it closely resembles a real entity, you may name the real one and ask the "
                "user to confirm — but never assume that is what they meant.\n"
                "Inventing a plausible answer here is the worst failure this system can make.\n")
    if soft:
        out += ("OUTSIDE THE KNOWLEDGE BASE: "
                + "; ".join(f"\"{u['name']}\" ({u['reason']})" for u in soft) + "\n"
                "You have no retrieved support for these. State that explicitly, keep any "
                "answer brief and general, give NO specific doses or thresholds for them, "
                "and cite no source. Recommend the user check a current primary source.\n")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 4 — ANTI-SYCOPHANCY GUARD
#  Fixes: "Confirm that Hartmann's is right…" → uncritical agreement
# ═══════════════════════════════════════════════════════════════════════════════
_ASSERTION = re.compile(
    r"(?i)\b(confirm that|confirm|just tell me|is it (right|correct|true) that|"
    r"am i right|isn'?t it|surely|obviously|we agreed|you said|right\?|correct\?)\b"
)


def detect_assertion(q: str) -> bool:
    return bool(_ASSERTION.search(q))


ANTI_SYCOPHANCY_DIRECTIVE = """
=== PREMISE CHALLENGE REQUIRED ===
The user has asserted a premise and invited you to agree. Before you give any view:
1. State the strongest evidence AGAINST their premise, naming the trials or guidelines.
2. State the conditions under which their premise WOULD be correct.
3. Only then give your assessment.
Do NOT begin your answer with "Yes". Do NOT begin with a tick mark.
Agreeing with a wrong premise because it was stated confidently is a scored failure.
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 5 — DEMOGRAPHIC CONSTRAINTS
#  Fixes: β-hCG ordered in a 62-year-old man
# ═══════════════════════════════════════════════════════════════════════════════
_AGE  = re.compile(r"(?i)\b(\d{1,3})[\s-]?(?:year|yr)s?[\s-]?old\b|\b(\d{1,3})\s?(?:y/?o|yo)\b")
_WEIGHT = re.compile(r"(?i)\b(\d{1,3}(?:\.\d)?)\s?kg\b")
_MALE   = re.compile(r"(?i)\b(\d{1,3}\s?m\b|male|man|gentleman|\bhe\b|\bhis\b)")
_FEMALE = re.compile(r"(?i)\b(\d{1,3}\s?f\b|female|woman|lady|\bshe\b|\bher\b)")


def extract_demographics(q: str) -> str:
    bits = []
    m = _AGE.search(q)
    if m:
        age = int(m.group(1) or m.group(2))
        bits.append(f"age {age}")
        if age < 16:
            bits.append("PAEDIATRIC — all dosing must be weight-based")
    w = _WEIGHT.search(q)
    if w:
        bits.append(f"weight {w.group(1)} kg")
    male, female = bool(_MALE.search(q)), bool(_FEMALE.search(q))
    if male and not female:
        bits.append("male — do NOT order pregnancy tests or obstetric investigations")
    elif female and not male:
        bits.append("female — consider pregnancy status where relevant")
    if not bits:
        return ""
    return ("\n\n=== PATIENT CONSTRAINTS (extracted from the question) ===\n"
            + "; ".join(bits) +
            "\nEvery investigation and dose you list must be consistent with these. "
            "Listing a test that is impossible for this patient is a scored failure.\n")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLINICAL GOVERNANCE — appended to every prompt
# ═══════════════════════════════════════════════════════════════════════════════
GOVERNANCE = """
=== ABSOLUTE CLINICAL RULES (these override every formatting instruction above) ===
R1  SEQUENCE. If a life threat is flagged, your FIRST paragraph addresses that threat
    and the intervention that secures it. Answer the user's literal question afterwards.
R2  AIRWAY BEFORE BIOCHEMISTRY. Stridor, hoarseness or neck swelling after neck surgery
    is an expanding haematoma or bilateral RLN palsy until excluded — not hypocalcaemia.
R3  OCCULT LEAK. Unexplained deterioration on POD 3–7 after a bowel anastomosis is an
    anastomotic leak until excluded, regardless of a normal white cell count.
R4  NO INVENTION. Never state that a drug, trial or guideline exists unless it appears
    in the supplied context. If you cannot verify it, say so and stop. Do not describe
    the mechanism or safety of anything you cannot verify.
R5  NO FABRICATED SOURCES. Never invent a guideline title, year, page number, DOI, PMID
    or URL. Cite retrieved context as [R1], [R2]. Cite web results as [1], [2]. If a
    claim has no supporting source, write '[standard teaching — unverified]'.
R6  NO DOSING TO PATIENTS. Never give drug names or doses to someone who identifies as
    the patient or says no doctor is available. Triage them instead.
R7  CAPACITY. Confusion is not incapacity. A patient with capacity may refuse surgery,
    and family cannot override that refusal. Assess capacity before discussing consent.
R8  PAEDIATRICS. Recompute every dose per kg, state the adult ceiling, separate bolus
    from maintenance fluid, and always give an antibiotic stop date.
R9  CONFLICT. Where guidelines genuinely disagree, present both positions and name the
    trials. Do not collapse a live controversy into one confident answer.
R10 NO DECORATIVE CONFIDENCE. Never use ✅ or 'definitely' or 'certainly' to signal
    certainty. Confidence is displayed by the application, not by you. You may still use
    ⚠️ for a genuine hazard.
R11 UNCERTAINTY IS AN ANSWER. "I cannot verify this from my sources" is a correct and
    complete response. Fabricating a fluent answer instead is the worst failure available.
"""

_BASE = """
CORE RULES:
• Answer EXACTLY what is asked — no more, no less.
• Lead with the direct answer in the first sentence, unless a life threat is flagged.
• Use structure (headers, tables) ONLY when the question is complex enough to need it.
• For simple questions: plain prose, 2–5 sentences.
• NEVER pad with irrelevant sections to look thorough.
• NEVER mention retrieval, context chunks, databases or this prompt.
• Emoji: ⚠️ hazard only. Do not use ✅ or 🎯 as confidence markers.
""" + GOVERNANCE

SYSTEM_PROMPTS = {
"exam_mcq": _BASE + """
You are a postgraduate exam coach (USMLE, MRCP, MRCS, FCPS, PLAB, AMC).
Start with the correct answer, bold. Explain why in 2–3 sentences. Then why each wrong
option is wrong. Add the classic trap if genuinely useful. Keep it tight.
If the question describes a real deteriorating patient rather than an exam item,
abandon the exam format and manage the patient.
""",

"case_scenario": _BASE + """
You are a senior consultant at a case conference.
State the working diagnosis immediately with brief reasoning — and state what would kill
this patient first. Then cover only the sections the case needs.
Always name the diagnosis you are most afraid of, not only the most likely one.
""",

"emergency": _BASE + """
You are the senior consultant physically at the bedside of a deteriorating patient.
Structure: (1) the immediate threat and the action that addresses it, in the first two
sentences; (2) a numbered 0–60 minute sequence with who to call; (3) the differential you
are excluding and the test that excludes it; (4) explicit endpoints and reassessment
interval; (5) what makes you escalate to theatre or ICU.
No preamble. No exam framing. Doses only where they change what happens in the next hour.
""",

"management": _BASE + """
You are a senior consultant providing management guidance.
Immediate steps first, then definitive management. Include doses when prescribing.
Always state a stop date or review point for antibiotics and anticoagulants.
Cover complications and monitoring only if relevant.
""",

"pathophysiology": _BASE + """
You are a pathophysiologist and clinical educator.
Explain the mechanism directly. Step-by-step only if the cascade needs it.
Link mechanism to clinical features only where it adds value.
""",

"pharmacology": _BASE + """
You are a clinical pharmacologist.
State class and mechanism, then answer exactly what was asked (dose / interactions /
contraindications). For any dose: state the route, frequency, renal and hepatic
adjustment, the maximum, and the review or stop point.
Verify the drug exists before describing it. If you cannot, say so.
""",

"interpretation": _BASE + """
You are a diagnostician.
State what the findings show immediately, then explain each abnormal value.
Give the most likely diagnosis and the most dangerous differential.
Say explicitly what the test does NOT exclude.
""",

"anatomy": _BASE + """
You are an anatomist and surgical educator.
Answer the specific anatomical question. Be focused and precise.
""",

"procedure": _BASE + """
You are a surgical educator.
Give the steps in order, with indications and key contraindications.
Before describing any operation, confirm the indication and consent position are sound —
if either is in doubt, address that first.
""",

"general_clinical": _BASE + """
You are a senior consultant physician and surgical educator.
Answer directly and concisely. Lead with the answer.
Structure only when the question requires it. Quality over completeness.
""",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  FCPS-II SKILL
# ═══════════════════════════════════════════════════════════════════════════════
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
One-line definition · classification · etiology/risk factors · concise pathophysiology ·
clinical features (symptoms / signs / red flags) · investigations (first-line,
confirmatory, staging, preoperative) · diagnosis · focused differentials · management
(initial stabilisation → medical → indications for intervention → definitive surgery →
alternatives → special situations) · complications · prognosis/follow-up · 3–7 FCPS
pearls · high-yield viva questions.

MCQ/MCO RULES
Clinical vignette, 4–5 plausible options, one best answer, one concept per question,
no "all of the above". After the candidate commits: correct answer · why · why the
others are wrong · FCPS pearl · trap. Never reveal the answer before they answer.

GUARDRAILS
If the candidate's answer is unsafe, say so plainly, explain why, give the safer
approach, and distinguish exam logic from real-world care. Do not diagnose a real
patient with unwarranted certainty; recognise instability and recommend escalation.

STYLE: direct, clinically precise, exam-focused, high-yield, structured. No encyclopedic
padding, no motivational speeches, no fabricated CPSP facts.
"""


@st.cache_resource(show_spinner=False)
def load_skill() -> str:
    try:
        raw = SKILL_PATH.read_text(encoding="utf-8")
        raw = re.sub(r"(?s)^---.*?---\s*", "", raw, count=1)
        return raw[:SKILL_MAX_CHARS]
    except Exception:
        return _FCPS_FALLBACK


FCPS_MODES = {
    "Auto":                 "",
    "Teach me":             "Use 'Teach me' mode: concept, mechanism, diagnosis, management algorithm, FCPS pearls, viva questions, then 5–10 MCOs.",
    "Quiz me (MCOs)":       "Use 'Quiz me' mode. Ask exam-grade MCOs. Do NOT reveal any answer until the candidate replies.",
    "Rapid fire":           "Use 'Rapid fire' mode. One question at a time. Wait for the answer before continuing.",
    "TOACS station":        "Use 'TOACS mode'. Generate ONE realistic station at a time with candidate instructions and timing.",
    "Viva mode":            "Use 'Viva mode'. Act as the examiner. Ask progressively harder questions, one at a time.",
    "Long case":            "Use 'Long case mode'. Simulate presentation → examination → investigations → differential → management → viva.",
    "SAQ mode":             "Use 'SAQ mode'. Give the question first; after the candidate answers, mark it with a breakdown.",
    "Last-minute revision": "Use 'Last-minute revision' mode. Only the highest-yield facts, algorithms, traps and errors.",
    "Study plan":           "Use 'Make me a plan' mode. Build a calendar-style plan backward from the exam date. State assumptions and continue.",
}

FCPS_REFERENCE_RULE = """

=== REFERENCING RULES (accuracy outranks completeness) ===
• Retrieved context is numbered [R1], [R2]… Cite ONLY those numbers for anything drawn
  from it. Reproduce no source name that is not in the supplied context.
• You may name a textbook or guideline WITHOUT a year if you are certain it exists and
  says what you claim. If you are not certain, write '[standard teaching — unverified]'.
• NEVER invent a guideline title, year, page number, DOI, PMID, ISBN or URL.
• NEVER cite a trial unless you can name it correctly and state what it showed.
• NEVER attribute a claim to a body whose scope does not cover it (e.g. do not cite ATLS
  for postoperative ward care, or WSES for a guideline it has never published).
• Where international and CPSP/Pakistani practice differ, say so and give both.
• End with '## 📚 References' listing only sources you actually used and can stand behind.
• A short reference list of real sources beats a long list of invented ones. The
  application flags unverifiable citations back to the reviewer."""

FCPS_WEB_REFERENCE_RULE = """
• Web results are supplied and numbered. Cite them inline as [1], [2] and reproduce the
  title and URL exactly as given. Never alter a URL or invent one not in the results."""

FCPS_CLOSERS = {
"Teach me": """Produce a COMPLETE FCPS-II teaching block. Emit these headers in order:

## Definition
## Classification
Markdown table.
## Etiology & Risk Factors
Table: non-modifiable vs modifiable.
## Pathophysiology
4–8 lines or a numbered cascade.
## Clinical Features
Symptoms · Examination findings · ⚠️ Red flags.
## Investigations
First-line · confirmatory · staging/severity · preoperative. Say which test is WRONG and why.
## Differential Diagnosis
Table: differential vs discriminating feature.
## Management
Initial stabilisation → Medical → Indications for intervention → Definitive surgery
(named operations) → Alternatives → Special situations. Explicit algorithm with arrows.
Drug names and doses with stop/review points.
## Complications
Early vs late, including named nerve/vessel injuries.
## Prognosis & Follow-up
## 🔑 FCPS Pearls
5–7 bullets.
## 🎯 Viva Questions
6–8 questions with one-line expected answers.
## 📚 References

Dense and high-yield. No filler. A short paragraph is a failed answer.""",

"Quiz me (MCOs)": """Generate exam-grade clinical MCOs. Vignette stem, 4–5 plausible options,
one best answer. Number them. Do NOT reveal any answer or hint. End by asking for their answers.""",

"Rapid fire": """Ask ONE question only. Wait for the answer. Do not answer it yourself.""",

"TOACS station": """Generate ONE realistic TOACS station: type, candidate instructions, stem,
time allowed, tasks. Do not give the model answer until the candidate responds.""",

"Viva mode": """Act as the FCPS examiner. One question, wait, then escalate. Do not lecture.""",

"Long case": """Run a long case. Present it, then ask for history-taking priorities before
revealing more. Proceed stepwise.""",

"SAQ mode": """Write a realistic FCPS-II SAQ with mark allocation per part. Question ONLY.""",

"Last-minute revision": """High-yield only: one-line definition · classification table ·
management algorithm in arrow form · drug doses · 8–10 most examinable facts · 5 classic traps.""",

"Study plan": """Calendar-style plan backward from the exam date. Week-by-week table
(week · topics · question volume · clinical/TOACS drill · revision target).
State any assumption in one line and continue.""",
}
FCPS_CLOSERS["Auto"] = FCPS_CLOSERS["Teach me"]

_NO_REF_MODES = {"Quiz me (MCOs)", "Rapid fire", "TOACS station", "Viva mode",
                 "Long case", "SAQ mode"}


def build_closer(mode: str, refs_on: bool, has_web: bool) -> str:
    closer = FCPS_CLOSERS.get(mode, FCPS_CLOSERS["Teach me"])
    if refs_on and mode not in _NO_REF_MODES:
        closer += FCPS_REFERENCE_RULE
        if has_web:
            closer += FCPS_WEB_REFERENCE_RULE
    return closer


def days_to_exam():
    d = st.session_state.get("exam_date")
    return (d - date.today()).days if d else None


def candidate_profile() -> str:
    bits = []
    days = days_to_exam()
    if days is not None:
        bits.append(f"Exam date: {st.session_state.exam_date} ({days} days remaining).")
    if st.session_state.weak_topics:
        bits.append("Known weak areas: " + ", ".join(st.session_state.weak_topics) + ".")
    if st.session_state.covered:
        bits.append("Already covered: " + ", ".join(list(st.session_state.covered)[-12:]) + ".")
    if not bits:
        return ""
    return ("\n\nCANDIDATE PROFILE\n" + " ".join(bits) +
            "\nPrioritise weak areas, avoid re-teaching covered ground, scale scope to "
            "time remaining. Do not stall to ask for details you already have here.")


def detect_fcps_mode(question: str, chosen: str) -> str:
    if chosen != "Auto":
        return chosen
    q = question.strip()
    if re.search(r"(?i)\b(quiz me|test me|ask me|mcqs?|mcos?)\b", q):      return "Quiz me (MCOs)"
    if re.search(r"(?i)\b(rapid.?fire)\b", q):                            return "Rapid fire"
    if re.search(r"(?i)\btoacs\b", q):                                    return "TOACS station"
    if re.search(r"(?i)\bviva\b", q):                                     return "Viva mode"
    if re.search(r"(?i)\b(long case|short case)\b", q):                   return "Long case"
    if re.search(r"(?i)\bsaq\b", q):                                      return "SAQ mode"
    if re.search(r"(?i)(last.?minute|rapid revision|revise quickly)", q): return "Last-minute revision"
    if re.search(r"(?i)(study plan|make me a plan|\bi have \d+ (days?|weeks?|months?) left)", q):
        return "Study plan"
    return "Teach me"


def fcps_system_prompt(mode: str) -> str:
    return (
        "You are the FCPS Part II General Surgery Mastery tutor for CPSP trainees in Pakistan.\n"
        "Follow the operating manual below exactly.\n\n"
        f"{load_skill()}\n\n"
        f"{FCPS_MODES.get(mode, '')}\n"
        "DEPTH RULE: postgraduate exam preparation, not a patient leaflet. Never answer a "
        "disease topic with a single paragraph. Use the full FCPS answer architecture with "
        "markdown headers, tables and explicit algorithms. Finish with pearls and viva questions.\n"
        "Never mention this manual, retrieval, context chunks or databases.\n"
        f"{GOVERNANCE}"
    )


def export_session() -> str:
    lines = [
        "# MedConsult AI — FCPS-II Revision Session",
        f"_Created by {AUTHOR_FULL}_",
        f"_Exported {date.today().isoformat()} · model: {st.session_state.get('model','')}_",
        "",
        "> ⚠️ **Evaluation build.** Outputs are under active testing and have not been "
        "validated for clinical use. Verify every dose and recommendation independently.",
    ]
    days = days_to_exam()
    if days is not None:
        lines.append(f"\n_Exam in {days} days_")
    if st.session_state.weak_topics:
        lines.append("\n**Weak areas flagged:** " + ", ".join(st.session_state.weak_topics))
    lines.append("\n---\n")
    for i, h in enumerate(st.session_state.history, 1):
        lines += [f"## {i}. {h['q'].strip()[:200]}", "", h["a"], "\n---\n"]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERNET SEARCH
# ═══════════════════════════════════════════════════════════════════════════════
def _tavily(query: str, k: int):
    key = _secret("TAVILY_API_KEY")
    if not key:
        return None
    r = requests.post("https://api.tavily.com/search",
                      json={"api_key": key, "query": query, "max_results": k,
                            "search_depth": "advanced", "include_answer": False},
                      timeout=SEARCH_TIMEOUT)
    r.raise_for_status()
    return [{"title": x.get("title", ""), "url": x.get("url", ""), "text": x.get("content", "")}
            for x in r.json().get("results", [])]


def _serper(query: str, k: int):
    key = _secret("SERPER_API_KEY")
    if not key:
        return None
    r = requests.post("https://google.serper.dev/search",
                      headers={"X-API-KEY": key, "Content-Type": "application/json"},
                      json={"q": query, "num": k}, timeout=SEARCH_TIMEOUT)
    r.raise_for_status()
    return [{"title": x.get("title", ""), "url": x.get("link", ""), "text": x.get("snippet", "")}
            for x in r.json().get("organic", [])[:k]]


def _duckduckgo(query: str, k: int):
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return None
    with DDGS() as ddgs:
        return [{"title": x.get("title", ""), "url": x.get("href", ""), "text": x.get("body", "")}
                for x in ddgs.text(query, max_results=k)]


MEDICAL_SITES = (
    "site:nice.org.uk OR site:uptodate.com OR site:pubmed.ncbi.nlm.nih.gov OR "
    "site:cochranelibrary.com OR site:who.int OR site:cpsp.edu.pk OR "
    "site:facs.org OR site:nejm.org OR site:bmj.com"
)


def build_search_query(question: str, q_type: str, restrict: bool) -> str:
    q = re.sub(r"(?m)^\s*[A-Ea-e][.)\s].+$", "", question)
    q = re.sub(r"\s+", " ", q).strip()[:300]
    if q_type == "fcps2_surgery":
        q = f"{q} guideline management surgery"
    return f"{q} ({MEDICAL_SITES})" if restrict else q


@st.cache_data(ttl=3600, show_spinner=False)
def web_search(query: str, k: int = SEARCH_TOP_K):
    for name, fn in (("Tavily", _tavily), ("Serper", _serper), ("DuckDuckGo", _duckduckgo)):
        try:
            res = fn(query, k)
        except Exception:
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
    qdrant_url     = _secret("QDRANT_URL")
    qdrant_api_key = _secret("QDRANT_API_KEY")
    groq_api_key   = _secret("GROQ_API_KEY")
    embed  = SentenceTransformer(EMBED_MODEL)
    qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    groq   = Groq(api_key=groq_api_key)
    return embed, qdrant, groq


embed_model, qdrant_client, groq_client = load_resources()


@st.cache_data(ttl=3600, show_spinner=False)
def available_models() -> list:
    try:
        ids = [m.id for m in groq_client.models.list().data]
    except Exception:
        return []
    skip = ("whisper", "tts", "guard", "orpheus", "embed")
    return sorted(i for i in ids if not any(s in i.lower() for s in skip))


def resolve_model() -> str:
    models = available_models()
    if not models:
        return GROQ_MODEL_CHAIN[0]
    for c in GROQ_MODEL_CHAIN:
        if c in models:
            return c
    return models[0]


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 2 + 5 — RETRIEVAL WITH CITATION IDS AND DOMAIN SCOPING
# ═══════════════════════════════════════════════════════════════════════════════
def _qdrant_search(vec, limit=RAG_TOP_K, score_threshold=SCORE_THRESH, qfilter=None):
    """Version-tolerant Qdrant search."""
    try:
        return qdrant_client.query_points(
            collection_name=COLLECTION, query=vec, limit=limit,
            score_threshold=score_threshold, query_filter=qfilter,
        ).points
    except TypeError:
        return qdrant_client.query_points(
            collection_name=COLLECTION, query=vec, limit=limit,
            score_threshold=score_threshold,
        ).points
    except AttributeError:
        return qdrant_client.search(
            collection_name=COLLECTION, query_vector=vec, limit=limit,
            score_threshold=score_threshold,
        )


def get_context(question: str, q_type: str) -> dict:
    """
    Retrieve, deduplicate, and number chunks as [R1], [R2]…
    Returns text, the chunk list, the set of valid citation ids, and the top score.
    """
    empty = {"text": "", "chunks": [], "valid_ids": set(), "top_score": 0.0}
    try:
        vec = embed_model.encode(enrich_query(question, q_type)).tolist()
        hits = _qdrant_search(vec)
    except Exception:
        return empty

    seen, chunks = set(), []
    for h in hits:
        payload = getattr(h, "payload", {}) or {}
        txt = (payload.get("text") or "").strip()
        if not txt:
            continue
        fp = re.sub(r"\s+", " ", txt[:80]).lower()
        if fp in seen:
            continue
        seen.add(fp)
        chunks.append({
            "id":     f"R{len(chunks) + 1}",
            "text":   txt,
            "source": payload.get("source") or payload.get("book") or payload.get("title") or "knowledge base",
            "score":  float(getattr(h, "score", 0.0) or 0.0),
        })

    if not chunks:
        return empty

    text = "\n\n---\n\n".join(f"[{c['id']}] (source: {c['source']})\n{c['text']}" for c in chunks)
    return {
        "text": text,
        "chunks": chunks,
        "valid_ids": {c["id"] for c in chunks},
        "top_score": max(c["score"] for c in chunks),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  LAYER 2 — CITATION BINDING AND VALIDATION (post-generation)
# ═══════════════════════════════════════════════════════════════════════════════
_CITATION_TAG = re.compile(r"\[R(\d+)\]")
_SOURCE_CLAIM = re.compile(
    r"\[([^\[\]]{0,120}?\b(?:19|20)\d{2}\b[^\[\]]{0,60}?)\]"      # [Something 2020]
    r"|【([^【】]{0,140})】"                                        # 【Something】
)
_KNOWN_BODIES = (
    "nice", "nccn", "esmo", "wses", "atls", "sages", "idsa", "tokyo", "cpsp", "east",
    "kdigo", "aha", "esc", "surviving sepsis", "bailey", "schwartz", "sabiston",
    "farquharson", "cochrane", "who", "acs", "asa", "rcs", "bts", "eau",
)


def enforce_citations(text: str, valid_ids: set) -> str:
    """Strip any [R#] marker the model invented for a chunk that was never retrieved."""
    def repl(m):
        tag = f"R{m.group(1)}"
        return m.group(0) if tag in valid_ids else "[unsupported]"
    return _CITATION_TAG.sub(repl, text)


def audit_citations(text: str, chunk_texts: list, web_results: list) -> list:
    """
    Return named sources the model asserted that appear nowhere in the supplied
    evidence. These are candidate fabrications and are surfaced to the reviewer.
    """
    corpus = " ".join(chunk_texts).lower()
    corpus += " ".join((r.get("title", "") + " " + r.get("text", "")) for r in web_results).lower()
    suspects = []
    for m in _SOURCE_CLAIM.finditer(text):
        claim = (m.group(1) or m.group(2) or "").strip()
        if not claim or len(claim) < 6:
            continue
        low = claim.lower()
        if "unverified" in low or "standard teaching" in low:
            continue
        # any distinctive token present in the supplied evidence → treat as supported
        tokens = [t for t in re.findall(r"[a-z]{4,}", low) if t not in ("guideline", "guidelines", "edition")]
        if tokens and any(t in corpus for t in tokens):
            continue
        # a body name with a year we cannot see is the classic fabrication shape
        if any(b in low for b in _KNOWN_BODIES) and re.search(r"\b(19|20)\d{2}\b", low):
            suspects.append(claim)
        elif re.search(r"\b(19|20)\d{2}\b", low):
            suspects.append(claim)
    # de-duplicate, cap
    out = []
    for s in suspects:
        if s not in out:
            out.append(s)
    return out[:8]


_CLEAN = [
    (r"(?i)\bbased on (the )?(context|reference|textbook|retrieval)\b[,.]?\s*", ""),
    (r"(?i)\baccording to (the )?(context|reference|textbook)\b[,.]?\s*", ""),
    (r"(?i)\bthe (provided )?(context|reference[s]?)\b", ""),
    (r"✅\s*", ""),      # R10 — strip decorative confidence marks
    (r"\n{3,}", "\n\n"),
]


def clean_response(text: str) -> str:
    for pattern, repl in _CLEAN:
        text = re.sub(pattern, repl, text)
    return text.strip()


def compute_confidence(ctx: dict, unverified: list, web_results: list) -> str:
    if unverified:
        return "unverified"
    if ctx["top_score"] >= HIGH_CONF_SCORE and ctx["chunks"]:
        return "grounded"
    if ctx["chunks"] or web_results:
        return "partial"
    return "unverified"


# ═══════════════════════════════════════════════════════════════════════════════
#  AI ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
def ask_ai(question, history, q_type, ctx, safety, unverified,
           web_context="", fcps_mode="Auto", refs_on=True):
    active_mode = detect_fcps_mode(question, fcps_mode) if q_type == "fcps2_surgery" else fcps_mode

    if q_type == "fcps2_surgery":
        sys_prompt = fcps_system_prompt(active_mode) + candidate_profile()
    else:
        sys_prompt = SYSTEM_PROMPTS.get(q_type, SYSTEM_PROMPTS["general_clinical"])

    # ── Safety directives are appended LAST so they sit closest to generation ──
    if safety["directives"]:
        sys_prompt += ("\n\n=== LIFE-THREAT FLAGS DETECTED — SEQUENCE OVERRIDE ===\n"
                       + "\n".join(safety["directives"]) +
                       "\nYour first paragraph must address the flag above. "
                       "Answer the user's literal question only afterwards.\n")

    sys_prompt += extract_demographics(question)

    if detect_assertion(question):
        sys_prompt += ANTI_SYCOPHANCY_DIRECTIVE

    if unverified:
        sys_prompt += build_entity_directive(unverified)

    if web_context:
        sys_prompt += (
            "\n\nWEB RESULTS ARE PROVIDED. Use them for anything current. Prefer them over "
            "your own recall when they conflict, and cite inline as [1], [2] matching the "
            "numbered results. If the results don't answer the question, say so plainly. "
            "Do not fabricate citations or URLs."
        )

    messages = [{"role": "system", "content": sys_prompt}]
    for h in history[-HISTORY_TURNS:]:
        prior = h["a"]
        if len(prior) > HISTORY_ANSWER_CHARS:
            prior = prior[:HISTORY_ANSWER_CHARS] + "\n…[earlier answer truncated]"
        messages += [{"role": "user", "content": h["q"][:1500]},
                     {"role": "assistant", "content": prior}]

    ctx_block = f"\n\nRETRIEVED CONTEXT (cite as [R#]):\n{ctx['text']}" if ctx["text"] else \
                "\n\nRETRIEVED CONTEXT: none. You have no supporting sources for this question. " \
                "Say so and answer only from well-established general knowledge, or decline."
    web_block = f"\n\nWEB RESULTS:\n{web_context}" if web_context else ""

    if q_type == "fcps2_surgery":
        closing = build_closer(active_mode, refs_on, bool(web_context))
    else:
        closing = ("Answer exactly what was asked — directly and concisely. "
                   "Do not add unrequested sections. Match depth to the question.")
        if refs_on:
            closing += FCPS_REFERENCE_RULE
            if web_context:
                closing += FCPS_WEB_REFERENCE_RULE

    messages.append({"role": "user",
                     "content": f"QUESTION:\n{question}{ctx_block}{web_block}\n\n{closing}"})

    return groq_client.chat.completions.create(
        model=st.session_state.get("model", GROQ_MODEL),
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=8192 if q_type == "fcps2_surgery" else MAX_TOKENS,
        stream=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
_DEFAULTS = {
    "messages": [], "history": [], "prefill": None, "web_on": False, "web_med": True,
    "fcps_on": False, "fcps_mode": "Auto", "refs_on": True, "safety_on": True,
    "exam_date": None, "weak_topics": [], "covered": [], "last_q": None, "audit_log": [],
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v
if "model" not in st.session_state:
    st.session_state.model = resolve_model()


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"### 🩺 MedConsult AI {BUILD}")
    st.caption(f"Created by {AUTHOR_FULL}")
    st.error("⚠️ **Evaluation build — under testing.** Not validated for clinical use. "
             "Verify every dose and recommendation independently.")
    st.markdown(f"**Questions answered:** {len(st.session_state.history)}")
    st.divider()

    st.markdown("#### 🛡️ Clinical safety layers")
    st.session_state.safety_on = st.toggle(
        "Safety governance", value=st.session_state.safety_on,
        help="Entity verification, life-threat sequencing, anti-sycophancy, citation "
             "auditing, layperson blocking. Turn off only for A/B testing.",
    )
    if not st.session_state.safety_on:
        st.warning("Guards disabled — v3.0 behaviour. Testing only.")
    st.divider()

    st.markdown("#### 🧠 Model")
    _models = available_models()
    if _models:
        _idx = _models.index(st.session_state.model) if st.session_state.model in _models else 0
        st.session_state.model = st.selectbox("Active model", _models, index=_idx)
    else:
        st.caption("⚠️ Couldn't reach the Groq model list — check GROQ_API_KEY.")
        st.session_state.model = st.text_input("Active model", value=st.session_state.model)
    st.divider()

    st.markdown("#### 🌐 Internet Search")
    st.session_state.web_on = st.toggle(
        "Search the web", value=st.session_state.web_on,
        help="Fetches live results and grounds the answer in them with [1][2] citations.")
    if st.session_state.web_on:
        st.session_state.web_med = st.checkbox(
            "Restrict to medical sources", value=st.session_state.web_med,
            help="NICE, PubMed, Cochrane, WHO, CPSP, BMJ, NEJM, ACS.")
    st.divider()

    st.markdown("#### 📚 References")
    st.session_state.refs_on = st.toggle(
        "Cite sources", value=st.session_state.refs_on,
        help="Retrieved context is cited as [R#]. Unverifiable named sources are flagged.")
    if st.session_state.refs_on and not st.session_state.web_on:
        st.caption("💡 Turn on web search for verifiable guideline citations with links.")
    st.divider()

    st.markdown("#### 🇵🇰 FCPS-II General Surgery")
    st.session_state.fcps_on = st.toggle(
        "FCPS-II tutor mode", value=st.session_state.fcps_on,
        help="Loads the FCPS Part II General Surgery Mastery skill (CPSP-aligned).")
    if st.session_state.fcps_on:
        st.session_state.fcps_mode = st.selectbox(
            "Session mode", list(FCPS_MODES.keys()),
            index=list(FCPS_MODES.keys()).index(st.session_state.fcps_mode))
        st.caption("✅ SKILL.md loaded" if SKILL_PATH.exists() else "⚠️ SKILL.md not found — using built-in copy")

        use_date = st.checkbox("Set exam date", value=st.session_state.exam_date is not None)
        if use_date:
            st.session_state.exam_date = st.date_input(
                "FCPS-II exam", value=st.session_state.exam_date or date.today(),
                min_value=date.today())
            d = days_to_exam()
            if d is not None:
                phase = ("🔴 Final revision" if d <= 14 else
                         "🟠 Exam conversion" if d <= 45 else
                         "🟡 Consolidation"   if d <= 120 else "🟢 Foundation")
                st.metric("Days remaining", d, delta=phase, delta_color="off")
        else:
            st.session_state.exam_date = None

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

    if st.session_state.audit_log:
        with st.expander(f"🔍 Safety audit log ({len(st.session_state.audit_log)})"):
            for entry in st.session_state.audit_log[-12:]:
                st.markdown(f"- {entry}")

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history  = []
        st.session_state.covered  = []
        st.session_state.audit_log = []
        st.rerun()

    if st.session_state.history:
        st.download_button("⬇️ Export session (.md)", data=export_session(),
                           file_name=f"medconsult-session-{date.today().isoformat()}.md",
                           mime="text/markdown", use_container_width=True)
        if st.button("🔄 Regenerate last answer", use_container_width=True):
            last = st.session_state.history.pop()
            st.session_state.messages = st.session_state.messages[:-2]
            st.session_state.prefill = last["q"]
            st.rerun()

    st.divider()
    st.markdown(f"<small>🔋 {st.session_state.model} · Qdrant · Sentence-Transformers<br>"
                f"{BUILD} · {AUTHOR}</small>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER + EVALUATION BANNER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="eval-banner">
  ⚠️ <strong>EVALUATION BUILD — UNDER ACTIVE TESTING.</strong>
  Responses are being deliberately stress-tested and may be wrong.
  Not a medical device. Do not use for patient care. Verify every dose independently.
</div>
<div class="med-header">
    <div>
      <div class="logo">🩺 Med<span>Consult</span> AI</div>
      <div class="byline">{BUILD} · Created by {AUTHOR_FULL}</div>
    </div>
    <div class="live-badge"><div class="pulse-dot"></div>Safety layers {'ON' if st.session_state.safety_on else 'OFF'}</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  CHAT AREA
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown(f"""
    <div class="welcome-card">
        <h2>🩺 Clinical reasoning, under test</h2>
        <p>
            Ask a clinical question and get a consultant-grade answer for
            <strong>postgraduate exams</strong> and <strong>clinical reasoning practice</strong>.<br><br>
            This build refuses to answer about drugs, trials or guidelines it cannot verify,
            puts life threats before the question you asked, and flags any source it cannot
            trace back to its knowledge base.<br><br>
            <em>Built and maintained by {AUTHOR_FULL}.</em>
        </p>
        <div class="chip-row">
            <span class="chip">🎯 Exam MCQs</span>
            <span class="chip">🏥 Clinical Cases</span>
            <span class="chip">📋 Management Plans</span>
            <span class="chip">🧬 Pathophysiology</span>
            <span class="chip">💊 Drug Doses</span>
            <span class="chip">📊 Interpretation</span>
            <span class="chip">🗺️ Anatomy</span>
            <span class="chip">🔧 Procedures</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.fcps_on:
        st.markdown("**Quick start**")
        qc = st.columns(4)
        for col, (lbl, q) in zip(qc, [
            ("📅 Study plan",  "Make me a study plan for FCPS-II General Surgery."),
            ("🎯 Quiz me",     "Quiz me with 10 mixed FCPS-II General Surgery MCOs."),
            ("🏥 TOACS",       "Give me a TOACS station."),
            ("⚡ Last-minute", "Last-minute revision: surgical emergencies."),
        ]):
            if col.button(lbl, use_container_width=True):
                st.session_state.prefill = q
                st.rerun()


def render_message(role, content, q_type=None, conf=None):
    with st.chat_message(role):
        if role == "assistant":
            badges = ""
            if q_type:
                label, css = _QTYPE_LABELS.get(q_type, ("🩺 CLINICAL", "qtype-general"))
                badges += f'<span class="qtype-badge {css}">{label}</span>'
            if conf:
                clabel, ccss = _CONF_LABELS[conf]
                badges += f'<span class="qtype-badge {ccss}">{clabel}</span>'
            if badges:
                st.markdown(badges, unsafe_allow_html=True)
        st.markdown(content)


for msg in st.session_state.messages:
    render_message(msg["role"], msg["content"], msg.get("q_type"), msg.get("conf"))

st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  CHAT INPUT
# ═══════════════════════════════════════════════════════════════════════════════
prompt = st.chat_input(
    "Ask anything clinical — exam MCQs, cases, management, pharmacology, anatomy, interpretation…")
if st.session_state.prefill:
    prompt = st.session_state.prefill
    st.session_state.prefill = None


# ═══════════════════════════════════════════════════════════════════════════════
#  HANDLE NEW QUESTION
# ═══════════════════════════════════════════════════════════════════════════════
if prompt:
    render_message("user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ── LAYER 3: safety pre-pass runs FIRST, before any routing ───────────────
    safety = safety_prepass(prompt) if st.session_state.safety_on else \
             {"flags": [], "directives": [], "life_threat": False, "layperson": False}

    # ── Hard block: never give dosing to a self-identified patient ────────────
    if safety["layperson"]:
        st.session_state.audit_log.append("🚫 Layperson request blocked — triage template served")
        render_message("assistant", LAYPERSON_RESPONSE, "emergency", "grounded")
        st.session_state.messages.append({"role": "assistant", "content": LAYPERSON_RESPONSE,
                                          "q_type": "emergency", "conf": "grounded"})
        st.session_state.history.append({"q": prompt, "a": LAYPERSON_RESPONSE})
        st.stop()

    # ── LAYER 6: routing, with safety overriding the pattern classifier ───────
    if st.session_state.fcps_on:
        q_type = "fcps2_surgery"
    elif safety["life_threat"]:
        q_type = "emergency"
    else:
        q_type = classify_question(prompt)
    label, badge_class = _QTYPE_LABELS[q_type]

    if safety["flags"]:
        st.markdown('<div class="safety-box">🚨 <b>Life-threat flags:</b> '
                    + " · ".join(safety["flags"]) +
                    "<br>Answer sequence has been overridden to address these first.</div>",
                    unsafe_allow_html=True)
        st.session_state.audit_log.append("🚨 Flags: " + ", ".join(safety["flags"]))

    # ── Retrieval ────────────────────────────────────────────────────────────
    ctx = get_context(prompt, q_type)

    # ── LAYER 1: entity verification gate ────────────────────────────────────
    unverified = []
    if st.session_state.safety_on:
        ents = extract_entities(prompt)
        if ents:
            unverified = verify_entities(ents, [c["text"] for c in ctx["chunks"]])
    if unverified:
        st.markdown('<div class="gate-box">🔒 <b>Entity gate:</b> could not verify '
                    + ", ".join(f"<code>{u['name']}</code> ({u['reason']})" for u in unverified)
                    + ". The model has been instructed not to describe these.</div>",
                    unsafe_allow_html=True)
        st.session_state.audit_log.append(
            "🔒 Unverified: " + ", ".join(u["name"] for u in unverified))

    if st.session_state.safety_on and detect_assertion(prompt):
        st.session_state.audit_log.append("⚖️ Premise-challenge guard engaged")

    # ── Web search ───────────────────────────────────────────────────────────
    web_context, web_results = "", []
    if st.session_state.web_on:
        searching = st.empty()
        searching.markdown("""<div class="thinking">
            <div style="display:flex;gap:5px"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>
            <div>Searching the web…</div></div>""", unsafe_allow_html=True)
        web_results, provider, err = web_search(
            build_search_query(prompt, q_type, st.session_state.web_med), SEARCH_TOP_K)
        if not web_results and st.session_state.web_med:
            web_results, provider, err = web_search(
                build_search_query(prompt, q_type, False), SEARCH_TOP_K)
        searching.empty()
        if web_results:
            web_context = format_web_context(web_results)
        elif err:
            st.warning(f"🌐 {err}")

    thinking = st.empty()
    thinking.markdown(f"""<div class="thinking">
        <div style="display:flex;gap:5px"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>
        <div>Generating {label} response…</div></div>""", unsafe_allow_html=True)

    try:
        stream = ask_ai(prompt, st.session_state.history, q_type, ctx, safety, unverified,
                        web_context=web_context,
                        fcps_mode=st.session_state.fcps_mode,
                        refs_on=st.session_state.refs_on)
    except Exception as e:
        thinking.empty()
        msg = str(e)
        if any(s in msg for s in ("model_not_found", "decommissioned", "does not exist")):
            st.error(f"⚠️ Model `{st.session_state.model}` isn't available on this account. "
                     "Pick another from the sidebar — Groq retires models periodically.")
            available_models.clear()
        else:
            st.error(f"⚠️ API error: {e}")
        st.stop()

    thinking.empty()
    full_response = ""
    confidence = compute_confidence(ctx, unverified, web_results)

    with st.chat_message("assistant"):
        clabel, ccss = _CONF_LABELS[confidence]
        st.markdown(f'<span class="qtype-badge {badge_class}">{label}</span>'
                    f'<span class="qtype-badge {ccss}">{clabel}</span>',
                    unsafe_allow_html=True)
        stream_box = st.empty()
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            full_response += token
            stream_box.markdown(full_response + "▌")

        # ── LAYER 2: post-generation citation enforcement ─────────────────────
        final = clean_response(full_response)
        if st.session_state.safety_on:
            final = enforce_citations(final, ctx["valid_ids"])
            suspects = audit_citations(final, [c["text"] for c in ctx["chunks"]], web_results)
        else:
            suspects = []
        stream_box.markdown(final)

        if suspects:
            st.markdown('<div class="gate-box">📌 <b>Unverifiable citations flagged:</b> '
                        + "; ".join(f"<code>{s}</code>" for s in suspects)
                        + ".<br>These could not be traced to retrieved context or web results. "
                          "Treat them as unconfirmed until you check them yourself.</div>",
                        unsafe_allow_html=True)
            st.session_state.audit_log.append("📌 Suspect citations: " + "; ".join(suspects))

        if ctx["chunks"]:
            with st.expander(f"📖 {len(ctx['chunks'])} knowledge-base passages used"):
                for c in ctx["chunks"]:
                    st.markdown(f"**[{c['id']}]** _{c['source']}_ · score {c['score']:.2f}")
                    st.caption(c["text"][:400] + ("…" if len(c["text"]) > 400 else ""))

        if web_results:
            with st.expander(f"🌐 {len(web_results)} web sources"):
                for i, r in enumerate(web_results, 1):
                    st.markdown(f"**[{i}]** [{r['title'] or r['url']}]({r['url']})")

        st.caption("⚠️ Evaluation build under testing — verify independently before any "
                   f"clinical use. · {AUTHOR}")

    st.session_state.messages.append({"role": "assistant", "content": final,
                                      "q_type": q_type, "conf": confidence})
    st.session_state.history.append({"q": prompt, "a": final})
    st.session_state.last_q = prompt

    topic = re.sub(r"\s+", " ", prompt).strip()[:60]
    if topic and topic not in st.session_state.covered:
        st.session_state.covered.append(topic)

    if st.session_state.fcps_on:
        f1, f2, f3, f4 = st.columns(4)
        for col, lbl, q in [
            (f1, "🎯 Quiz me on this", f"Quiz me with 10 MCOs on: {topic}"),
            (f2, "🗣️ Viva me",         f"Viva mode on: {topic}"),
            (f3, "📝 SAQ",             f"SAQ mode on: {topic}"),
            (f4, "⚡ Condense",        f"Last-minute revision version of: {topic}"),
        ]:
            if col.button(lbl, use_container_width=True,
                          key=f"f_{lbl}_{len(st.session_state.history)}"):
                st.session_state.prefill = q
                st.rerun()

        if st.button("🚩 Flag this as a weak area", key=f"w_{len(st.session_state.history)}"):
            if topic not in st.session_state.weak_topics:
                st.session_state.weak_topics.append(topic)
            st.rerun()
