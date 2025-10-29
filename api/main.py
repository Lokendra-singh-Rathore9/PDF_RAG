import os
from typing import Any, Dict, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from logger import GLOBAL_LOGGER as log
from src.data_ingestion.ingestion import ChatIngestor
from src.data_retriever.retriever import ConversationalRAG
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

class ChatResponse(BaseModel):
    answer: str


# ===============================
# File Adapter for UploadFile
# ===============================
class FastAPIFileAdapter:
    """Adapts FastAPI UploadFile to behave like a normal file object."""
    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.name = uf.filename

    def read(self) -> bytes:
        """Read file content."""
        self._uf.file.seek(0)
        return self._uf.file.read()

    def getbuffer(self) -> bytes:
        """Compatibility for .getbuffer()"""
        return self.read()


# ----------------------------
# Simple in-memory chat history
# ----------------------------
SESSIONS: Dict[str, List[dict]] = {}


# ===============================
# FastAPI App Setup
# ===============================
app = FastAPI(title="Document Portal API", version="0.1")

BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# Routes
# ===============================
@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    """Serve HTML UI."""
    log.info("Serving UI homepage.")
    resp = templates.TemplateResponse("index.html", {"request": request})
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/health")
def health() -> Dict[str, str]:
    """Basic health check."""
    log.info("Health check passed.")
    return {"status": "ok", "service": "document-portal"}


# ===============================
# Upload & Index PDF
# ===============================
@app.post("/chat/upload_files")
async def chat_build_index(file: UploadFile = File(...)) -> Any:
    """Handle PDF upload and build vector index."""
    try:
        log.info(f"Uploading and indexing file: {file.filename}")

        # Wrap the FastAPI UploadFile
        wrapped = FastAPIFileAdapter(file)

        # Initialize the ChatIngestor
        ci = ChatIngestor()
        saved_path = ci.save_uploaded_files(wrapped)  # Save PDF locally

        retriever = ci.built_retriver()  # Build vector DB

        if retriever is None:
            raise HTTPException(status_code=500, detail="Retriever could not be built.")

        log.info(f"Index created successfully for session: {ci.session_id}")
        SESSIONS[ci.session_id] = []
        return {
            "status": "success",
            "session_id": ci.session_id,
            "file_path": str(saved_path)
        }

    except Exception as e:
        log.exception("Chat index building failed")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")


# ===============================
# Query Chat (Retrieve Answers)
# ===============================
@app.post("/chat/query")
async def chat_query(
    session_id: str = Form(...),
    query: str = Form(...),
) -> Any:
    
    if not session_id or session_id not in SESSIONS:
        raise HTTPException(status_code=400, detail="Invalid or expired session_id. Re-upload documents.")
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    """Retrieve similar documents and return answer."""
    try:
        log.info(f"Query received: {query} (session={session_id})")

        # Build RAG and load retriever from persisted FAISS with MMR
        rag = ConversationalRAG(session_id=session_id)
        index_path = f"data/vectorstores/{session_id}"
        rag.load_retriever_from_faiss(
            index_path=index_path,
        )

        # Use simple in-memory history and convert to BaseMessage list
        simple = SESSIONS.get(session_id, [])
        lc_history = []
        for m in simple:
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                lc_history.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_history.append(AIMessage(content=content))

        answer = rag.invoke(query, chat_history=lc_history)

        # Update history
        simple.append({"role": "user", "content": query})
        simple.append({"role": "assistant", "content": answer})
        SESSIONS[session_id] = simple
        print(answer)
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")
# uvicorn api.main:app --port 8080 --reload    
