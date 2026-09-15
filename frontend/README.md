# DocuMind Frontend

A minimal React interface for the DocuMind API. Two screens only:

- **Documents** — upload, list, and delete documents.
- **Chat** — ask questions about one selected document and see answers with page sources.

No feature exists here that isn't backed by a real endpoint.

## Endpoint mapping

| UI action | Endpoint |
|---|---|
| Upload document | `POST /api/v1/documents` |
| Documents table | `GET /api/v1/documents` |
| Delete button | `DELETE /api/v1/documents/{id}` |
| Chat send | `POST /api/v1/documents/{id}/chat` |
| Connection dot | `GET /api/v1/health` |

`GET /api/v1/documents/{id}` exists in the API but isn't called directly by this UI — the documents list already carries everything the screens need.

## Setup

```bash
npm install
npm run dev
```

By default the app calls the backend at `http://localhost:8000/api/v1`. To point it elsewhere, create a `.env` file:

```
VITE_API_BASE_URL=http://your-backend-host/api/v1
```

## Notes

- Uploads are synchronous: the request stays open until ingestion finishes, matching the backend's synchronous `/documents` endpoint. The button shows "Uploading…" for the duration.
- The chat screen blocks sending a question until the document's status is `completed`, mirroring the check the backend itself performs.
- Chat history is kept in memory only, per browser session — there's no backend endpoint to persist it.
