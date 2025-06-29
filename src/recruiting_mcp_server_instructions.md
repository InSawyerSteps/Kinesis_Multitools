# Recruiting MCP Server

This document explains what the **Recruiting MCP Server** is, the tools/end-points it exposes, and how to run it locally.

---
## 1. Overview
`recruiting_mcp_server.py` is a standalone FastAPI application that exposes recruiting-specific functionality originally contained inside `toolz.py`.  It focuses on **resume & job-description analysis** and uses:

* **FastAPI** – HTTP layer
* **ChromaDB** – persistent vector store for resume embeddings
* **Ollama** – local embedding generator (`nomic-embed-text` model)
* **thefuzz** – (optional) fuzzy matching for hybrid search

The server lives in the same `src/` directory as the original MCP server but listens on **port 8001** by default.

---
## 2. Prerequisites
| Requirement | Notes |
|-------------|-------|
| Python ≥ 3.10 | Install via pyenv, asdf, or system package manager |
| Virtualenv / Poetry | Recommended to keep dependencies isolated |
| Ollama  | Must be running locally; `ollama serve` should expose <http://localhost:11434/> |
| Git / make | Helpful for typical workflow |

**Python packages** are listed in `requirements.txt`. If you have not installed them yet:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---
## 3. Running the Server
```bash
# From the repo root
uvicorn src.recruiting_mcp_server:app --host 0.0.0.0 --port 8001 --reload
```
* `--reload` auto-reloads on code changes (development only).
* Logs are printed to the console via the `logging` module.

Once running, open the interactive docs:
```
http://localhost:8001/docs
```

---
## 4. End-points

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/index-resume` | Chunk & embed a resume file into ChromaDB |
| POST | `/semantic-search-resumes` | Semantic similarity search over indexed resumes |
| POST | `/hybrid-search` | Hybrid ranking: semantic + tech keyword / alias matching |
| POST | `/parse-jd` | Extract structure from a job description file or raw text |
| POST | `/parse-resume` | Extract text & structured info from a resume |

### 4.1. Example – Index a Resume
```jsonc
POST /index-resume
{
  "project_name": "RecruitingDemo",
  "file_path": "C:/Projects/RecruitingDemo/resumes/alice_smith.pdf"
}
```
Response:
```jsonc
{
  "status": "success",
  "chunks_indexed": 12,
  "embedding_errors": []
}
```

### 4.2. Example – Hybrid Search
```jsonc
POST /hybrid-search
{
  "query": "Senior backend engineer",
  "core_technologies": ["Python", "FastAPI", "AWS"],
  "max_results": 5
}
```

---
## 5. Data Locations
* **Resumes root** per project configured in `PROJECT_RESUME_PATHS` inside the server script.
* **ChromaDB persistent store** defaults to `<repo_root>/chroma_db_data`. Safe to delete to clear all vectors (server recreates automatically).

---
## 6. Troubleshooting Checklist
1. **Ollama not running?** – `curl http://localhost:11434` should return a JSON banner.
2. **ChromaDB permission errors?** – Ensure the process can create/modify `chroma_db_data` directory.
3. **Missing optional packages** – End-points that need `thefuzz`, `docx`, or `pdfminer` will return a descriptive 400 error.
4. **CORS** – Enable in uvicorn or FastAPI `app.add_middleware` if calling from a browser.

---
## 7. Production Notes (Optional)
* Run behind **gunicorn + uvicorn workers** for concurrency.
* Persist `chroma_db_data` on durable storage.
* Secure endpoints with an auth layer (e.g., API key or OAuth) if exposed publicly.
