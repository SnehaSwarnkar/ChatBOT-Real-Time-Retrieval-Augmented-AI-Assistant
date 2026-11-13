# ChatBOT — Real-Time Retrieval-Augmented AI Assistant

A fully local, privacy-first AI chatbot that answers questions from your own documents using **Retrieval-Augmented Generation (RAG)**. No cloud API calls — everything runs on your machine via [Ollama](https://ollama.com).

---

## Features

- **Hybrid retrieval** — combines vector search (ChromaDB) + BM25 keyword search + Reciprocal Rank Fusion (RRF) for high-quality document retrieval
- **Cross-encoder re-ranking** — uses a sentence-transformer cross-encoder to re-rank candidates before answering
- **Multi-format support** — ingest PDF and PowerPoint (PPTX) files
- **Multimodal uploads** — upload PDF or image files (PNG, JPG, JPEG, WEBP) directly in chat for instant Q&A
- **Conversation history** — remembers context across turns with automatic query rewriting for follow-up questions
- **RAG evaluation suite** — built-in scripts for Faithfulness, Answer Relevance, Precision@K, Recall@K, and MRR
- **React frontend** — clean chat UI with sidebar, file attachments, source citations, and chat history
- **FastAPI backend** — REST API with CORS, file upload handling, and health check endpoint
- **100% local** — all LLM inference runs through Ollama; no data leaves your machine

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Ollama (`llama3.2`) |
| Vision model | Ollama (`llava:7b`) |
| Embeddings | Ollama (`nomic-embed-text`) |
| Vector store | ChromaDB (persistent) |
| Keyword search | BM25 (`rank_bm25`) |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Backend | FastAPI + Uvicorn |
| Frontend | React 18 + Vite 5 |

---

## Project Structure

```
rag-chatbot/
├── main.py               # Document ingestion pipeline (PDF + PPTX → ChromaDB)
├── chat.py               # RAG pipeline: hybrid retrieval + re-ranking + answer generation
├── server.py             # FastAPI server (chat and file upload endpoints)
├── evaluation.py         # Retrieval evaluation: Precision@K, Recall@K, MRR
├── faithfulness.py       # Faithfulness evaluation (claim-level LLM grading)
├── answer_relevance.py   # Answer relevance evaluation (LLM-as-judge scoring)
├── requirements.txt      # Python dependencies
├── data/                 # Place your PDF and PPTX files here for ingestion
└── frontend/
    ├── src/
    │   ├── App.jsx       # Main React chat interface
    │   └── App.css       # Styles
    ├── package.json
    └── vite.config.js
```

---

## Prerequisites

1. **Python 3.10+**
2. **Node.js 18+**
3. **Ollama** installed and running — [download here](https://ollama.com/download)

Pull the required models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
ollama pull llava:7b
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/SnehaSwarnkar/ChatBOT-Real-Time-Retrieval-Augmented-AI-Assistant.git
cd ChatBOT-Real-Time-Retrieval-Augmented-AI-Assistant
```

### 2. Create a Python virtual environment

```bash
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

---

## Usage

### Step 1 — Ingest your documents

Place PDF or PPTX files in the `data/` folder, then run:

```bash
python main.py
```

This extracts text, chunks it, generates embeddings via Ollama, and stores everything in ChromaDB.

### Step 2 — Start the FastAPI backend

```bash
uvicorn server:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at `http://127.0.0.1:8000`. Interactive docs at `http://127.0.0.1:8000/docs`.

### Step 3 — Start the frontend

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Root — server info |
| `GET` | `/api/health` | Health check + document count |
| `POST` | `/api/chat` | Text-based RAG chat |
| `POST` | `/api/chat/upload` | Chat with uploaded file (PDF / image) |

---

## Evaluation

Run the evaluation scripts after ingesting documents and starting the backend:

```bash
# Retrieval metrics (Precision@K, Recall@K, MRR)
python evaluation.py

# Faithfulness (are answers grounded in retrieved context?)
python faithfulness.py

# Answer relevance (does the answer address the question?)
python answer_relevance.py
```

---

## Environment Variables

Create a `.env` file in the project root if you need to override defaults:

```
VISION_MODEL=llava:7b
```

The `.env` file is listed in `.gitignore` and will never be committed.

---

## Notes

- The `data/` folder ships with sample PPTX files for demo purposes. Replace them with your own documents.
- ChromaDB is stored locally in `chroma_db_ollama/` — this directory is excluded from git.
- The conversation history is per-server-process (in-memory). Restarting the server resets history.
