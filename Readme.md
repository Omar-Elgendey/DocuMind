<div align="center">

# DocuMind

**Upload a document. Ask questions. Get answers grounded in its content.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel-black?style=for-the-badge&logo=vercel)](https://docu-mind-lac.vercel.app/)
[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

</div>

---

## Overview

DocuMind is an AI-powered Retrieval-Augmented Generation (RAG) application that allows users to upload documents and ask questions about their content.

Instead of relying only on an LLM's general knowledge, DocuMind retrieves relevant pieces of the uploaded document and provides them to the model as context before generating a response. This approach helps keep answers grounded in the source document and reduces unsupported responses.

**Core flow:**

```
Upload Document
      ↓
Extract Text
      ↓
Split into Chunks
      ↓
Generate Embeddings
      ↓
Store in ChromaDB
      ↓
Retrieve Relevant Chunks
      ↓
Build Prompt
      ↓
Generate Answer with LLM
```

---

## Live Demo

**Try DocuMind:** [https://docu-mind-lac.vercel.app/](https://docu-mind-lac.vercel.app/)

No account is required to use the application. The current version supports document upload, processing, document management, and document-based question answering.

---

## Demo

### Document Q&A

Users can upload a document and ask natural-language questions about its content.

**Example workflow:**

```
Upload Document
      ↓
Document Processing
      ↓
Status: Completed
      ↓
Ask a Question
      ↓
Generate Grounded Response
```
###  Screenshots & Demo

![App Screenshot](./docs/chat.png)

<video src="./docs/screen.mp4" controls width="100%"></video>

## Features

### Document Processing
- Upload PDF, DOCX, PPTX, and TXT documents
- Extract text from uploaded files
- Split extracted text into overlapping chunks
- Generate multilingual vector embeddings
- Persist document vectors in ChromaDB

### RAG Question Answering
- Ask natural-language questions about uploaded documents
- Retrieve semantically relevant document chunks
- Pass retrieved content to the LLM as context
- Generate answers grounded in the retrieved document content
- Instruct the model to avoid unsupported answers when the required information is not available

### Document Management
- List uploaded documents
- View individual document information
- Track document processing status
- Delete documents and their associated vector data

### Session Isolation
- Each browser session receives its own session identifier
- Document operations are scoped using the `X-Session-ID` header
- Users only access documents associated with their current session

### Modular Architecture
The RAG system is separated into independent components:
- Loaders
- Text splitter
- Embedding provider
- Vector store
- Retriever
- Prompt builder
- LLM provider
- RAG pipeline
- Document service

This keeps the core AI pipeline independently testable and easier to extend.

---

## Architecture

DocuMind uses a **modular monolith** architecture.

### Document Ingestion

```
                         DOCUMENT INGESTION

┌──────────────┐
│    Upload    │
│ PDF/DOCX/    │
│ PPTX/TXT     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    Loader    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Splitter   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Embeddings  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   ChromaDB   │
│ Vector Store │
└──────────────┘

       │
       └──────────────► MySQL
                        Document Metadata
```

### Question Answering

```
                         QUESTION ANSWERING

┌──────────────┐
│   Question   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Retriever   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Relevant    │
│   Chunks     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    Prompt    │
│  + Context   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│     LLM      │
│    (Groq)    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    Answer    │
└──────────────┘
```

### RAG Pipeline

The complete pipeline consists of the following stages:

1. **Loader** — Extracts text from the uploaded document based on its file type.
2. **Text Splitter** — Splits extracted text into smaller overlapping chunks suitable for semantic retrieval and LLM context.
3. **Embeddings** — Each chunk is converted into a vector representation using `intfloat/multilingual-e5-small`. The multilingual embedding model allows the system to work with content across multiple languages.
4. **ChromaDB** — The generated vectors are stored in a persistent ChromaDB collection together with document and session metadata.
5. **Retriever** — When a user asks a question, the query is embedded and compared against stored document chunks to retrieve the most relevant content.
6. **Prompt Construction** — The retrieved chunks are inserted into a structured prompt together with the user's question.
7. **LLM Generation** — The LLM generates the final response using the retrieved document content as context.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Python 3.11+ |
| RAG Framework | LangChain |
| Vector Store | ChromaDB |
| Embeddings | Hugging Face — `intfloat/multilingual-e5-small` |
| LLM | Groq API |
| Database | MySQL |
| ORM | SQLAlchemy |
| Frontend | React |
| Containerization | Docker |
| Backend Deployment | Railway |
| Frontend Deployment | Vercel |
| Testing | Pytest, unittest.mock |

---

## Project Structure

```
DocuMind/
│
├── app/
│   ├── api/
│   │   └── routes/
│   │       └── documents.py
│   │
│   ├── db/
│   │   ├── models.py
│   │   ├── repository.py
│   │   └── session.py
│   │
│   ├── rag/
│   │   ├── loaders.py
│   │   ├── splitter.py
│   │   ├── embedding.py
│   │   ├── vector_store.py
│   │   ├── retriever.py
│   │   ├── prompt.py
│   │   ├── llm.py
│   │   └── pipeline.py
│   │
│   └── services/
│       └── document_service.py
│
├── tests/
│
├── frontend/
│
├── data/
│   └── uploads/
│
├── Dockerfile
├── pytest.ini
├── init_db.py
└── README.md
```

---

## Design Decisions

### Why RAG?

Sending an entire document directly to an LLM is not always practical. RAG allows DocuMind to:
- Process the document once
- Store its content as searchable vector representations
- Retrieve only the most relevant chunks for each question
- Provide those chunks to the LLM as context

This creates a more targeted document-questioning workflow.

### Why ChromaDB?

ChromaDB provides a lightweight vector database suitable for the project's local-first architecture while supporting persistent vector storage and semantic similarity search.

### Why MySQL + ChromaDB?

The two databases serve different purposes:

```
MySQL       →  Document metadata, processing state, document information
ChromaDB    →  Document chunks, embeddings, vector similarity search
```

Keeping these responsibilities separate makes the system easier to reason about and extend.

### Why a Modular Monolith?

DocuMind keeps its components inside a single application while maintaining clear boundaries between:

```
API → Service Layer → RAG Pipeline → Infrastructure
```

This avoids unnecessary distributed-system complexity while keeping the architecture modular.

---

## API Endpoints

All document endpoints use the `X-Session-ID` header to scope requests to the current browser session.

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/documents` | Upload and process a document |
| `GET` | `/documents` | List documents for the current session |
| `GET` | `/documents/{document_id}` | Get document details |
| `DELETE` | `/documents/{document_id}` | Delete a document and its associated vectors |
| `POST` | `/documents/{document_id}/chat` | Ask a question about a document |

**Example Chat Request**

```http
POST /documents/{document_id}/chat
X-Session-ID: <session-id>
Content-Type: application/json

{
  "question": "What is this document about?"
}
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- MySQL database
- Groq API key
- Git
- Docker (optional)

### Clone the Repository

```bash
git clone https://github.com/Omar-Elgendey/DocuMind.git
cd DocuMind
```

### Backend Setup

Create a virtual environment:

```bash
python -m venv venv
```

Activate it — on Windows:

```bash
venv\Scripts\activate
```

Or on Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install .
```

Create the environment file:

```bash
cp .env.example .env
```

Configure the required environment variables (see below), then initialize the database:

```bash
python init_db.py
```

Run the API:

```bash
uvicorn app.main:app --reload
```

The backend will be available at `http://localhost:8000`.

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### Running with Docker

Build the image:

```bash
docker build -t documind .
```

Run the backend:

```bash
docker run -p 8000:8000 --env-file .env documind
```

For the complete multi-service deployment, use the project's Docker configuration where available.

---

## Environment Variables

Create a `.env` file in the project root:

```env
# LLM Provider
GROQ_API_KEY=your_groq_api_key

# Database
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_HOST=your_db_host
DB_PORT=3306
DB_NAME=documind

# Vector Store
CHROMA_PERSIST_DIRECTORY=./data/chroma
CHROMA_COLLECTION_NAME=documind_collection

# Embedding Model
EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-small
```

---

## Testing

DocuMind includes unit tests for the core RAG and application components.

The test suite uses:
- Pytest
- unittest.mock

External services such as LLM APIs, databases, and other infrastructure dependencies are mocked where appropriate so tests do not require live external services.

Run the test suite with:

```bash
pytest
```

The tests cover:
- Normal operation
- Invalid inputs
- Edge cases
- Dependency failures
- Exception handling
- RAG component behavior

---

## Known Limitations

DocuMind was designed as a portfolio and learning project, so several production-level features are intentionally outside the current scope.

### Session-Based Isolation

The application uses browser sessions rather than full user authentication. Documents are scoped using the `X-Session-ID` header. This provides application-level separation between sessions but is **not** a complete security boundary.

A production implementation would introduce proper authentication and authorization, such as:
- User accounts
- JWT/session authentication
- User-owned documents
- Access-control checks

### Session Persistence

Clearing browser storage or switching devices creates a new session, so previously uploaded documents may no longer be accessible from that browser session.

### Current Scope

The current version focuses on the core document-RAG workflow rather than:
- User authentication
- Multi-user account management
- Background task queues
- Advanced document parsing
- Production-scale distributed infrastructure

---

## Future Improvements

Possible extensions include:
- User authentication and authorization
- Persistent user accounts
- More advanced document parsing
- Hybrid keyword + semantic retrieval
- Reranking retrieved chunks
- Conversation memory
- Background document processing
- Streaming LLM responses
- Improved source/citation visualization
- Additional document formats such as XLSX
- Production observability and monitoring

---

## Author

**Omar Elgendey**

- GitHub: [@Omar-Elgendey](https://github.com/Omar-Elgendey)
- LinkedIn: [omar-elgendey](https://www.linkedin.com/in/omar-elgendey-b142a0357)

---

## License

This project is licensed under the [MIT License](LICENSE).