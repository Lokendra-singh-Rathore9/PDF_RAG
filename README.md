# PDF_RAG

PDF_RAG enables Retrieval-Augmented Generation (RAG) workflows for PDF documents, letting you upload, process, and query PDFs using advanced language models and semantic search. It includes feedback mechanisms and evaluation tools for continuous improvement, supports API access with Groq integration, and provides a simple Streamlit UI for demonstration.

---

## 📹 Demo Video
Check out our demo video for the LLM-powered Text Classification API:\


https://github.com/user-attachments/assets/3701645a-f43a-4119-b076-0edf0d75198e



<br>
<details>
<summary>Or download: PDF_RAG.Demo.mp4</summary>
<a href="PDF_RAG.Demo.mp4">PDF_RAG.Demo.mp4</a>
</details>

---

## 🚀 Features

- Upload and process PDF documents
- Query PDFs with LLM-powered semantic search (Groq API integration)
- Collect user feedback to improve results
- Evaluation harness for RAG performance
- FastAPI endpoints for API access
- Streamlit UI for interactive demo

---

## 🛠️ Setup & Installation

### 1. Clone the repo

```bash
git clone https://github.com/Lokendra-singh-Rathore9/PDF_RAG.git
cd PDF_RAG
```

### 2. Install `uv` (Recommended)

```bash
# Install uv globally if not already done
pipx install uv
```

### 3. Create & activate a virtual environment

```bash
#In cmd terminal of folder
#Create & activate a virtual environment with Python 3.10
uv venv .venv
# Activate the environment
source .venv/bin/activate      # Linux/Mac
# or
.venv\Scripts\activate         # Windows PowerShell
```

### 4. Install dependencies

```bash
uv add -r requirements.txt
```

### 5. Configure environment variables

- Create `.env` file in root folder and set the required keys (such as API keys, DB paths, etc.), including your Groq API key:

```env
GROQ_API_KEY=<your_groq_api_key_here>
```

---

## ▶️ Run the API

```bash
uvicorn api.main:app --port 8080 --reload    

## this will take time if not start the try again after clossing the uvicorn command by ctrl+c and try again with above command
```



API available at: [http://localhost:8000](http://localhost:8000)

---

## 📚 API Endpoints

Below are the main endpoints used in PDF_RAG:

### `POST /upload_pdf`
Upload a PDF document for processing.

**Request (multipart/form-data):**
```
file: (PDF file)
```
**Response:**
```json
{
  "pdf_id": "unique_pdf_identifier",
  "filename": "your_file.pdf",
  "message": "PDF uploaded and processed successfully."
}
```

---

### `POST /query_pdf`
Query a PDF for information using semantic search with Groq LLM.

**Request:**
```json
{
  "pdf_id": "unique_pdf_identifier",
  "question": "What is the summary of this document?"
}
```
**Response:**
```json
{
  "answer": "The summary is ...",
  "context": ["Relevant context from PDF"],
  "latency_ms": 340
}
```

---
### `GET /healthz`
Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

---

## 🎯 How it Works

1. **PDF Upload:** Documents are ingested and indexed.
2. **Semantic Query:** LLM-powered search retrieves relevant information.
3. **Feedback Loop:** User feedback is collected for continuous improvement.
4. **Evaluation:** Built-in tools for measuring retrieval and answer quality.

---

## 🏗️ Project Structure
```
PDF_RAG/
├── .gitignore
├── .python-version
├── README.md
├── main.py
├── pyproject.toml
├── requirements.txt
├── uv.lock
├── api/              # API-related code (main.py)
├── logger/           # Logging utilities (Custom logger)
├── logs/             # Log files (Logs)
├── src/              # Main source code (data_ingestion, data_retriver file, prompts)
├── static/           # Static assets such as CSS, images (directory, contents not listed)
├── templates/        # HTML templates (directory, contents not listed)
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 👩‍💻 Tech Stack

- **Python** (60.7%)
- **CSS** (24.5%)
- **HTML** (14.8%)

---

## 🙏 Acknowledgments

Thanks to all contributors and the open-source community.


