import os
from typing import Any, Dict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from logger import GLOBAL_LOGGER as log
from src.data_ingestion.ingestion import ChatIngestor

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
    """Retrieve similar documents and return answer."""
    try:
        log.info(f"Query received: {query} (session={session_id})")

        ci = ChatIngestor()
        ci.session_id = session_id

        retriever = ci.built_retriver()
        if retriever is None:
            raise HTTPException(status_code=404, detail="Retriever not found or session invalid.")

        # Perform retrieval
        docs = retriever.get_relevant_documents(query)
        if not docs:
            return {"answer": "No relevant information found."}

        # Combine relevant results (you can later plug in an LLM here)
        answer = "\n\n".join([doc.page_content for doc in docs])

        log.info(f"Query answered successfully (session={session_id})")
        return {"answer": answer}

    except Exception as e:
        log.exception("Chat query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
# uvicorn api.main:app --port 8080 --reload    
