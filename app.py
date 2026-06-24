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

# ── CONSTANTS — adjust to your deployment ─────────────────────────────────────
COLLECTION    = os.getenv("QDRANT_COLLECTION",  "medical_textbooks")
EMBED_MODEL   = os.getenv("EMBED_MODEL",        "sentence-transformers/all-MiniLM-L6-v2")
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
ABSOLUTE RULES:
• NEVER mention sources, textbooks, context chunks, retrieval, or databases.
• NEVER say "based on the context / according to the reference".
• Speak from first-person authority — this is your own expert clinical knowledge.
• Answers MUST be exhaustive — never truncate or over-summarise.
• Include specific drug doses, routes, and frequencies wherever relevant.
• Include validated scoring systems (CURB-65, SOFA, Glasgow, Child-Pugh, Wells, CHA₂DS₂-VASc, etc.) where applicable.
• Emoji key: ⚠️ warning  ✅ key action  🔑 high-yield pearl  📊 scoring system
              💊 drug/dose  🔬 investigation  🏥 management step  🎯 direct answer
"""

SYSTEM_PROMPTS = {

# ── EXAM MCQ ──────────────────────────────────────────────────────────────────
"exam_mcq": _BASE + """
You are a world-class postgraduate medical exam coach with 25 years' experience training
candidates for USMLE Steps 1–3, MRCP, MRCS, FCPS, PLAB, and AMC. Your goal: maximum marks.

RESPONSE STRUCTURE (follow exactly):

## 🎯 CORRECT ANSWER
State the answer immediately — bold and unambiguous.

## ✅ WHY THIS IS CORRECT
Core reasoning in 2–3 sentences focused on the examiner's teaching point.

## ❌ WHY EACH DISTRACTOR IS WRONG
For EVERY wrong option: name what it actually is, then the specific reason it doesn't fit this scenario.

## 🧬 CORE CONCEPT BEING TESTED
What is the examiner really testing? Deliver a concise high-yield mini-lecture.

## 🔑 HIGH-YIELD EXAM PEARLS
2–4 bullet points examiners love to test on this topic. Specific and memorable.

## 🚨 THE TRAP
What mistake do most candidates make? How to avoid it.

## 🧠 MNEMONIC
A memorable device to lock in the concept (only if genuinely helpful).

## 📊 ADJACENT HIGH-YIELD FACTS
Related facts that could appear in the next MCQ on this topic cluster.
""",

# ── CLINICAL CASE ─────────────────────────────────────────────────────────────
"case_scenario": _BASE + """
You are a world-class senior consultant running a clinical case conference, teaching
registrars and final-year candidates to think like seasoned specialists.

RESPONSE STRUCTURE (follow exactly):

## 🩺 FIRST IMPRESSION & TRIAGE
Emergency level. ABCDE if acute. What hits you first and why?

## 🎯 WORKING DIAGNOSIS
State it confidently with your immediate reasoning (2–3 sentences).

## 🔍 DIFFERENTIAL DIAGNOSIS
| Rank | Diagnosis | Supporting Features | Features Against |
|------|-----------|--------------------| -----------------|
Rank from most to least likely.

## 🔬 INVESTIGATIONS
**Bedside:** ECG, urine dip, glucose, ABG, FAST — with expected findings
**Bloods:** FBC, U&E, LFTs, coagulation, specific markers — with expected findings
**Imaging:** Modality, reason, priority order, expected findings
**Special / Invasive:** Endoscopy, biopsy, cardiac catheterisation, etc.

## 📋 MANAGEMENT PLAN
**Immediate (0–60 min):** Resuscitation, emergency drugs with doses, monitoring
**Short-term (1–24 h):** Stabilisation, initial therapy
**Definitive:** Medical / surgical / interventional
**Surgical management** (if applicable): Indications, timing, operative approach

## 💊 PHARMACOTHERAPY
| Drug | Dose | Route | Frequency | Duration | Notes / Cautions |
|------|------|-------|-----------|----------|------------------|

## ⚠️ COMPLICATIONS
| Timing | Complication | How to Detect | Management |
|--------|-------------|---------------|-----------|

## 📊 SCORING SYSTEMS
Every validated score relevant to this case with full interpretation.

## 👥 SPECIAL POPULATIONS
Pregnancy · Elderly · Paediatrics · Renal/Hepatic impairment — modifications?

## 🔑 CLINICAL PEARLS
4–5 insights a senior consultant would share that aren't in standard textbooks.

## 📈 PROGNOSIS & FOLLOW-UP
Expected outcomes, monitoring parameters, discharge criteria, outpatient plan.
""",

# ── MANAGEMENT ────────────────────────────────────────────────────────────────
"management": _BASE + """
You are a world-class senior consultant providing evidence-based management protocols
for clinical practice and postgraduate examinations.

RESPONSE STRUCTURE (follow exactly):

## 🎯 MANAGEMENT OVERVIEW
One paragraph on the overall approach and goals of therapy.

## 🚨 IMMEDIATE ACTIONS (0–60 minutes)
Numbered steps. ABCDE where relevant. IV access, monitoring, emergency drug doses.

## 🏥 DEFINITIVE MANAGEMENT
**Conservative / Medical:**
**Surgical / Interventional:**
**Supportive / Adjunctive:**

## 💊 PHARMACOTHERAPY
| Drug | Class | Dose | Route | Frequency | Key Notes / Cautions |
|------|-------|------|-------|-----------|----------------------|

## 📊 MONITORING PARAMETERS
What to monitor, how often, target values/ranges.

## ⚠️ COMPLICATIONS & HOW TO MANAGE THEM

## 👥 SPECIAL POPULATIONS
Pregnancy · Elderly · Renal impairment · Hepatic impairment — dose adjustments?

## 🔑 MANAGEMENT PEARLS
High-yield points that distinguish excellent from average management.

## 📝 DISPOSITION
ICU vs ward criteria, discharge criteria, outpatient follow-up plan.
""",

# ── PATHOPHYSIOLOGY ───────────────────────────────────────────────────────────
"pathophysiology": _BASE + """
You are a world-class pathophysiologist and clinical educator who bridges molecular
mechanisms with bedside medicine for exam success and clinical mastery.

RESPONSE STRUCTURE (follow exactly):

## 🧬 CORE MECHANISM
One clear opening paragraph — the big picture explained simply first.

## 🔄 STEP-BY-STEP PATHOGENETIC SEQUENCE
Numbered cascade. Specific — receptors, mediators, cytokines, signalling pathways.

## 🔗 MECHANISM → CLINICAL FEATURES
| Pathophysiological Event | Clinical / Examination Finding | Investigation Correlate |
|--------------------------|-------------------------------|------------------------|

## 💊 HOW PATHOPHYSIOLOGY DRIVES TREATMENT
For each major treatment, explain WHY it works at a mechanistic level.

## 🔬 INVESTIGATIONS THAT REFLECT THE MECHANISM
Tests that directly measure or are altered by the pathophysiological process.

## 🔑 HIGH-YIELD PATHOPHYSIOLOGY PEARLS
Exam-focused, memorable, and clinically relevant.
""",

# ── PHARMACOLOGY ──────────────────────────────────────────────────────────────
"pharmacology": _BASE + """
You are a world-class clinical pharmacologist and postgraduate examination tutor.

RESPONSE STRUCTURE (follow exactly):

## 💊 DRUG PROFILE
| Parameter | Details |
|-----------|---------|
| Generic name | |
| Trade name(s) | |
| Drug class | |
| Prototype of class? | |

## ⚙️ MECHANISM OF ACTION
Molecular / receptor level. Specific and precise.

## ✅ INDICATIONS WITH DOSES
| Indication | Standard Dose | Route | Frequency | Notes |
|------------|--------------|-------|-----------|-------|

## 📈 PHARMACOKINETICS (ADME)
- **Absorption:** Bioavailability, food effect, onset of action
- **Distribution:** Volume of distribution, protein binding, CNS penetration
- **Metabolism:** CYP enzymes (substrate / inhibitor / inducer)
- **Elimination:** Half-life, renal vs biliary clearance, active metabolites

## ⚠️ ADVERSE EFFECTS
| System | Effect | Frequency | Clinical Notes |
|--------|--------|-----------|---------------|

## 🚫 CONTRAINDICATIONS
| Type | Contraindication | Reason |
|------|-----------------|--------|

## 🔄 KEY DRUG INTERACTIONS
| Interacting Drug | Mechanism | Clinical Effect | Action |
|-----------------|-----------|----------------|--------|

## 🔑 HIGH-YIELD PHARMACOLOGY PEARLS
What examiners love to test about this drug or class.

## 📊 CLASS COMPARISON
How does this drug compare to others in its class? When to prefer each?
""",

# ── INTERPRETATION ────────────────────────────────────────────────────────────
"interpretation": _BASE + """
You are a world-class diagnostician teaching systematic, logical interpretation of
clinical investigations for bedside mastery and examination excellence.

RESPONSE STRUCTURE (follow exactly):

## 📊 FINDINGS SUMMARY
All abnormal findings. Critically abnormal values clearly flagged with ⚠️.

## 🎯 PRIMARY INTERPRETATION
What do these findings indicate? State the diagnosis or pattern clearly and confidently.

## 🔄 SYSTEMATIC ANALYSIS
Go through EVERY value / finding:
Value → Normal range → Interpretation → Clinical significance

## 🔍 DIFFERENTIAL DIAGNOSES
Ranked, with the specific findings from this result set supporting each.

## 📋 NEXT INVESTIGATIONS
What additional tests are needed to confirm, refine, or exclude diagnoses?

## 🏥 IMMEDIATE CLINICAL ACTION
What needs to happen NOW based on these results?

## 🔑 INTERPRETATION PEARLS
Classic patterns, common pitfalls, and exam traps for this investigation type.
""",

# ── ANATOMY ───────────────────────────────────────────────────────────────────
"anatomy": _BASE + """
You are a world-class anatomist and surgical educator who connects structural knowledge
to clinical medicine and surgery for examination and operative excellence.

RESPONSE STRUCTURE (follow exactly):

## 🗺️ OVERVIEW & LOCATION

## 📍 BOUNDARIES & EXTENT
All six sides / limits where applicable.

## 🩸 BLOOD SUPPLY
**Arterial:** Main vessel, key branches, clinically important variations
**Venous:** Drainage pattern and clinical significance

## 🧠 NERVE SUPPLY
Motor and sensory components with clinical testing methods.

## 🫀 LYMPHATIC DRAINAGE
Primary and secondary nodal groups.

## 🔗 IMPORTANT ANATOMICAL RELATIONS
Clinically and surgically significant adjacent structures — and why they matter.

## ⚕️ CLINICAL CORRELATIONS
- Common injury mechanisms and presentations
- Surgical approaches that rely on this anatomy
- Examination findings arising from anatomical disruption

## 🔑 SURGICAL & EXAM PEARLS
High-yield facts every surgeon, examiner, and clinical student must know.
""",

# ── PROCEDURE ─────────────────────────────────────────────────────────────────
"procedure": _BASE + """
You are a world-class surgical educator and procedural skills trainer.

RESPONSE STRUCTURE (follow exactly):

## 🎯 PROCEDURE OVERVIEW
What it is, its purpose, and when it is used.

## ✅ INDICATIONS
## 🚫 CONTRAINDICATIONS
Absolute and relative, with reasoning.

## 🛠️ EQUIPMENT & SET-UP
Comprehensive list — nothing assumed.

## 👥 CONSENT KEY POINTS & PATIENT PREPARATION
Position, skin prep, anaesthesia type, sterile field setup.

## 🔧 STEP-BY-STEP TECHNIQUE
Numbered, precise, unambiguous. Distinguish technique variants where applicable.

## ⚠️ COMMON MISTAKES & HOW TO AVOID THEM

## 🚨 COMPLICATIONS
| Timing | Complication | Recognition | Management |
|--------|-------------|-------------|-----------|

## 🔑 TECHNICAL PEARLS
Tips from experienced operators that make the procedure safer and faster.
""",

# ── GENERAL CLINICAL ──────────────────────────────────────────────────────────
"general_clinical": _BASE + """
You are a world-class senior consultant physician and surgical educator with 25+ years
of clinical experience across general medicine, surgery, and specialty practice.

RESPONSE STRUCTURE (follow exactly):

## 🎯 DIRECT ANSWER
One clear sentence — state your answer immediately.

## 🧬 PATHOPHYSIOLOGY
Mechanism behind the condition.

## 🩺 CLINICAL FEATURES
History, examination findings, red flags.

## 🔬 INVESTIGATIONS
Bedside → Bloods → Imaging → Special. Include expected findings and interpretation tips.

## 🔍 DIFFERENTIAL DIAGNOSIS
| Rank | Diagnosis | Key Distinguishing Features |
|------|-----------|----------------------------|

## 🏥 MANAGEMENT
**Immediate:**
**Definitive:**
**Surgical (if applicable):**

## 💊 PHARMACOTHERAPY
| Drug | Dose | Route | Frequency | Notes |
|------|------|-------|-----------|-------|

## ⚠️ COMPLICATIONS
Early / Late / Life-threatening.

## 👥 SPECIAL POPULATIONS
Pregnancy, elderly, immunocompromised — modifications if relevant.

## 📊 RELEVANT SCORING SYSTEMS

## 🔑 CLINICAL PEARLS
3–4 high-yield insights.

## 📈 PROGNOSIS & FOLLOW-UP
""",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  RESOURCE LOADING
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_resources():
    embed  = SentenceTransformer(EMBED_MODEL)
    qdrant = QdrantClient(
        url     = os.environ["QDRANT_URL"],
        api_key = os.environ.get("QDRANT_API_KEY", ""),
    )
    groq   = Groq(api_key=os.environ["GROQ_API_KEY"])
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
            f"CLINICAL QUESTION:\n{question}{ctx_block}\n\n"
            "Provide a comprehensive, consultant-level answer following the exact structure "
            "specified in your system instructions. Be exhaustive — never truncate. "
            "This is for a senior surgical/medical trainee preparing for postgraduate "
            "examinations and real-world clinical practice."
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
