import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { chatWithDocument } from "../api.js";
import StatusBadge from "./StatusBadge.jsx";

export default function ChatPage({ document, onBack }) {
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(5);
  const [asking, setAsking] = useState(false);
  const [error, setError] = useState(null);

  const canChat = document.status === "completed";

  async function handleSubmit(event) {
    event.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || asking) return;

    setAsking(true);
    setError(null);

    try {
      const result = await chatWithDocument(document.id, trimmed, topK);
      setMessages((prev) => [
        ...prev,
        { question: trimmed, answer: result.answer, sources: result.sources },
      ]);
      setQuestion("");
    } catch (err) {
      setError(err.message);
    } finally {
      setAsking(false);
    }
  }

  return (
    <div style={{ maxWidth: "760px", margin: "0 auto", padding: "40px 24px" }}>
      <button
        onClick={onBack}
        style={{
          border: "none",
          background: "none",
          color: "var(--color-text-muted)",
          padding: 0,
          marginBottom: "16px",
          cursor: "pointer",
        }}
      >
        ← All documents
      </button>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "10px",
          marginBottom: "24px",
        }}
      >
        <h1 style={{ margin: 0, fontSize: "20px", fontWeight: 600 }}>
          {document.original_filename}
        </h1>
        <StatusBadge status={document.status} />
      </div>

      {!canChat && (
        <div
          style={{
            marginBottom: "20px",
            padding: "12px 14px",
            borderRadius: "var(--radius)",
            background: "var(--color-pending-bg)",
            color: "var(--color-pending-text)",
          }}
        >
          This document isn't ready for chat yet. Its status is "{document.status}".
        </div>
      )}

      <div
        style={{
          border: "1px solid var(--color-border)",
          borderRadius: "var(--radius)",
          background: "var(--color-surface)",
          padding: "20px",
          minHeight: "320px",
          marginBottom: "16px",
        }}
      >
        {messages.length === 0 ? (
          <p style={{ color: "var(--color-text-muted)" }}>
            Ask a question about this document to get started.
          </p>
        ) : (
          messages.map((message, index) => (
            <div
              key={index}
              style={{
                marginBottom: index === messages.length - 1 ? 0 : "24px",
              }}
            >
              {/* سؤال المستخدم */}
              <p
                style={{
                  margin: "0 0 12px",
                  padding: "8px 12px",
                  background: "var(--color-bg)",
                  borderRadius: "var(--radius)",
                  display: "inline-block",
                  fontWeight: 500,
                }}
              >
                {message.question}
              </p>

              {/* إجابة الـ RAG مع دعم Markdown والجداول */}
              <div
                style={{
                  margin: "0 0 12px",
                  lineHeight: 1.6,
                  fontSize: "14px",
                }}
              >
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {message.answer}
                </ReactMarkdown>
              </div>

              {/* مصادر الإجابة Chips */}
              {message.sources?.length > 0 && (
                <div
                  style={{
                    display: "flex",
                    flexWrap: "wrap",
                    gap: "6px",
                    marginTop: "8px",
                  }}
                >
                  {message.sources.map((source, sourceIndex) => (
                    <span
                      key={sourceIndex}
                      style={{
                        fontSize: "12px",
                        padding: "3px 8px",
                        borderRadius: "999px",
                        border: "1px solid var(--color-border)",
                        background: "var(--color-bg)",
                        color: "var(--color-text-muted)",
                      }}
                    >
                      {/* لو الصفحات مش متاحة بيظهر رقم الـ Chunk */}
                      {source.page !== null && source.page !== undefined
                        ? `Page ${source.page}`
                        : `Source ${sourceIndex + 1}`}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))
        )}
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

      <form onSubmit={handleSubmit}>
        <textarea
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Ask a question about this document…"
          rows={2}
          disabled={!canChat || asking}
          style={{
            width: "100%",
            padding: "10px 12px",
            border: "1px solid var(--color-border)",
            borderRadius: "var(--radius)",
            resize: "none",
            marginBottom: "10px",
          }}
        />
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justify: "space-between",
          }}
        >
          <label
            style={{
              display: "flex",
              alignItems: "center",
              gap: "6px",
              color: "var(--color-text-muted)",
            }}
          >
            top_k
            <input
              type="number"
              min={1}
              max={20}
              value={topK}
              onChange={(event) => setTopK(Number(event.target.value))}
              disabled={!canChat}
              style={{
                width: "48px",
                padding: "4px 6px",
                border: "1px solid var(--color-border)",
                borderRadius: "var(--radius)",
              }}
            />
          </label>
          <button
            type="submit"
            disabled={!canChat || asking || !question.trim()}
            style={{
              padding: "9px 18px",
              borderRadius: "var(--radius)",
              border: "1px solid var(--color-accent)",
              background: "var(--color-accent)",
              color: "#ffffff",
              fontWeight: 500,
              cursor: "pointer",
            }}
          >
            {asking ? "Asking…" : "Send"}
          </button>
        </div>
      </form>
    </div>
  );
}