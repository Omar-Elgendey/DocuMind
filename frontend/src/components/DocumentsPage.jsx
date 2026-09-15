import { useEffect, useRef, useState } from "react";
import { listDocuments, uploadDocument, deleteDocument } from "../api.js";
import StatusBadge from "./StatusBadge.jsx";
import ConfirmDialog from "./ConfirmDialog.jsx";

const SUPPORTED_EXTENSIONS = ".pdf,.docx,.pptx,.txt";

function formatDate(isoString) {
  if (!isoString) return "—";
  return new Date(isoString).toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

export default function DocumentsPage({ onOpenChat }) {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [pendingDelete, setPendingDelete] = useState(null);
  const [deleting, setDeleting] = useState(false);
  const fileInputRef = useRef(null);

  async function refresh() {
    setLoading(true);
    try {
      const data = await listDocuments();
      setDocuments(data);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function handleFileSelected(event) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;

    setUploading(true);
    setError(null);
    try {
      await uploadDocument(file);
      await refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  }

  async function handleConfirmDelete() {
    if (!pendingDelete) return;
    setDeleting(true);
    try {
      await deleteDocument(pendingDelete.id);
      setPendingDelete(null);
      await refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setDeleting(false);
    }
  }

  return (
    <div style={{ maxWidth: "960px", margin: "0 auto", padding: "40px 24px" }}>
      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: "24px" }}>
        <div>
          <h1 style={{ margin: 0, fontSize: "22px", fontWeight: 600 }}>Documents</h1>
          <p style={{ margin: "4px 0 0", color: "var(--color-text-muted)" }}>
            Upload a document, then chat with it once it finishes processing.
          </p>
        </div>

        <div>
          <input
            ref={fileInputRef}
            type="file"
            accept={SUPPORTED_EXTENSIONS}
            onChange={handleFileSelected}
            style={{ display: "none" }}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            style={{
              padding: "9px 16px",
              borderRadius: "var(--radius)",
              border: "1px solid var(--color-accent)",
              background: "var(--color-accent)",
              color: "#ffffff",
              fontWeight: 500,
            }}
          >
            {uploading ? "Uploading…" : "Upload document"}
          </button>
        </div>
      </div>

      {error && (
        <div
          style={{
            marginBottom: "16px",
            padding: "10px 14px",
            borderRadius: "var(--radius)",
            background: "var(--color-error-bg)",
            color: "var(--color-error-text)",
          }}
        >
          {error}
        </div>
      )}

      {uploading && (
        <div
          style={{
            marginBottom: "16px",
            padding: "10px 14px",
            borderRadius: "var(--radius)",
            border: "1px solid var(--color-border)",
            color: "var(--color-text-muted)",
          }}
        >
          Uploading and processing your document. This can take a moment for larger files.
        </div>
      )}

      <div
        style={{
          border: "1px solid var(--color-border)",
          borderRadius: "var(--radius)",
          background: "var(--color-surface)",
          overflow: "hidden",
        }}
      >
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ borderBottom: "1px solid var(--color-border)", textAlign: "left" }}>
              <Th>Filename</Th>
              <Th>Status</Th>
              <Th align="right">Chunks</Th>
              <Th>Uploaded</Th>
              <Th align="right">Actions</Th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={5} style={{ padding: "24px", textAlign: "center", color: "var(--color-text-muted)" }}>
                  Loading documents…
                </td>
              </tr>
            ) : documents.length === 0 ? (
              <tr>
                <td colSpan={5} style={{ padding: "24px", textAlign: "center", color: "var(--color-text-muted)" }}>
                  No documents yet. Upload one to get started.
                </td>
              </tr>
            ) : (
              documents.map((doc) => (
                <tr key={doc.id} style={{ borderBottom: "1px solid var(--color-border)" }}>
                  <Td>{doc.original_filename}</Td>
                  <Td>
                    <StatusBadge status={doc.status} />
                  </Td>
                  <Td align="right">{doc.chunks_count ?? "—"}</Td>
                  <Td>{formatDate(doc.created_at)}</Td>
                  <Td align="right">
                    <div style={{ display: "flex", justifyContent: "flex-end", gap: "8px" }}>
                      <button
                        onClick={() => onOpenChat(doc)}
                        style={{
                          padding: "6px 12px",
                          borderRadius: "var(--radius)",
                          border: "1px solid var(--color-border)",
                          background: "var(--color-surface)",
                        }}
                      >
                        Chat
                      </button>
                      <button
                        onClick={() => setPendingDelete(doc)}
                        style={{
                          padding: "6px 12px",
                          borderRadius: "var(--radius)",
                          border: "1px solid var(--color-border)",
                          background: "var(--color-surface)",
                          color: "var(--color-error-text)",
                        }}
                      >
                        Delete
                      </button>
                    </div>
                  </Td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {pendingDelete && (
        <ConfirmDialog
          title="Delete document?"
          message={`"${pendingDelete.original_filename}" will be permanently removed, along with its indexed content.`}
          confirmLabel="Delete"
          busy={deleting}
          onCancel={() => setPendingDelete(null)}
          onConfirm={handleConfirmDelete}
        />
      )}
    </div>
  );
}

function Th({ children, align = "left" }) {
  return (
    <th
      style={{
        padding: "10px 16px",
        fontSize: "12px",
        fontWeight: 600,
        color: "var(--color-text-muted)",
        textAlign: align,
      }}
    >
      {children}
    </th>
  );
}

function Td({ children, align = "left" }) {
  return (
    <td style={{ padding: "12px 16px", textAlign: align }}>{children}</td>
  );
}
