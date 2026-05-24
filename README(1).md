# 🏥 Medical AI Assistant — Deployment Guide

## Files
```
app.py                          ← main chatbot (deploy this)
step1_upload_to_cloud.py        ← run once on your PC
requirements.txt                ← Python dependencies
.streamlit/secrets.toml         ← API keys (DO NOT commit to GitHub)
```

---

## Step 1 — Get Free API Keys (5 minutes)

### Qdrant Cloud (vector database — free)
1. Go to https://qdrant.tech → Sign Up (free)
2. Create a cluster → copy the **URL** and **API Key**
3. Paste into `step1_upload_to_cloud.py`

### Groq (LLM — free, very fast)
1. Go to https://console.groq.com → Sign Up (free)
2. Create API Key → copy it

---

## Step 2 — Upload Your Books (run once on your PC)

```bash
pip install pypdf sentence-transformers qdrant-client
python step1_upload_to_cloud.py
```

This will:
- Read all PDFs from your books folder
- Create embeddings
- Upload everything to Qdrant Cloud

Takes 10–30 minutes depending on how many books you have.
You only need to do this ONCE (or when you add new books).

---

## Step 3 — Deploy to Streamlit Cloud (free)

1. Create a GitHub account at https://github.com
2. Create a new repository (e.g. `medical-ai`)
3. Upload these files:
   - `app.py`
   - `requirements.txt`
   (DO NOT upload secrets.toml or your PDFs)

4. Go to https://streamlit.io/cloud → Sign In with GitHub
5. Click **New App** → select your repo → select `app.py`
6. Go to **Advanced Settings → Secrets** and paste:

```toml
QDRANT_URL     = "https://xxxx.qdrant.tech"
QDRANT_API_KEY = "your-key-here"
GROQ_API_KEY   = "your-key-here"
```

7. Click **Deploy** — your app will be live in 2–3 minutes!

---

## Your Live App Will Have

- Beautiful dark medical UI
- Streaming answers (word by word)
- Conversation memory (follow-up questions work)
- Source citations (shows which book the answer came from)
- Sample question buttons
- List of all loaded books in sidebar
- Works on mobile and desktop

---

## Adding New Books Later

1. Add PDF to your `C:\Users\at444\Desktop\books` folder
2. Run `step1_upload_to_cloud.py` again
3. App auto-updates — no redeployment needed

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Collection not found" | Re-run step1_upload_to_cloud.py |
| Slow answers | Normal for first load; Groq is fast after warmup |
| Empty book list in sidebar | Check Qdrant API key in secrets |
