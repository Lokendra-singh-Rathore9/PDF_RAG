from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from logger import GLOBAL_LOGGER as log
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import uuid
from typing import List, Optional


class ChatIngestor:
    """Class to handle chat ingestion logic."""
    def __init__(self):
        def generate_session_id(prefix: str = "session") -> str:
            ist = ZoneInfo("Asia/Kolkata")
            return f"{prefix}_{datetime.now(ist).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        self.file_path: Path = Path()
        self.session_id: Optional[str] = generate_session_id()
        self.target_dir: Path = Path("data/pdf")
    

    def _split(self, docs: List[Document]) -> List[Document]:
        """Split PDF text into overlapping chunks."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        return chunks


    def save_uploaded_files(self, uploaded_files, target_dir: Path = Path("data/pdf")) -> Path:
        """Save uploaded files (Streamlit-like) and return local paths."""
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            uf = uploaded_files
            name = getattr(uf, "name", "file")
            ext = Path(name).suffix.lower()
            if ext != ".pdf":
                raise ValueError("Only PDF files are supported.")

            out = target_dir / Path(name).name
            with open(out, "wb") as f:
                data = uf.read()
                f.write(data)

            # Reset pointer (important if reused)
            if hasattr(uf, "seek"):
                uf.seek(0)

            self.file_path = out
            log.info(f"✅ File saved at: {out} ({out.stat().st_size} bytes)")
            return out
        except Exception as e:
            log.error(f"❌ Error saving uploaded file: {e}")
            raise


    def built_retriver(self):
        """Build retriever from ingested files."""
        if not self.file_path.exists():
            log.error(f"❌ File does not exist: {self.file_path}")
            return None

        if self.file_path.stat().st_size == 0:
            log.warning(f"⚠️ The file is empty: {self.file_path}")
            return None

        log.info(f"📄 Loading PDF from: {self.file_path}")
        loader = PyPDFLoader(str(self.file_path))
        documents = loader.load()
        log.info(f"✅ Documents loaded: {len(documents)} pages")

        chunks = self._split(documents)
        log.info(f"🧩 Split into {len(chunks)} text chunks")

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        log.info(f"💾 Vector store created with {len(chunks)} chunks")

        # Save vectorstore
        vectorstore_dir = Path("data/vectorstores")
        vectorstore_dir.mkdir(parents=True, exist_ok=True)
        vectorstore_path = vectorstore_dir / self.session_id
        vectorstore.save_local(str(vectorstore_path))
        log.info(f"✅ Vector store saved at: {vectorstore_path}")

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        log.info("🔍 Retriever is ready.")
        return retriever



