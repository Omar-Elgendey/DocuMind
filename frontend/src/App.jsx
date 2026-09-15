import { useEffect, useState } from "react";
import { checkHealth } from "./api.js";
import DocumentsPage from "./components/DocumentsPage.jsx";
import ChatPage from "./components/ChatPage.jsx";

export default function App() {
  const [activeDocument, setActiveDocument] = useState(null);
  const [connected, setConnected] = useState(null);

  useEffect(() => {
    checkHealth()
      .then(() => setConnected(true))
      .catch(() => setConnected(false));
  }, []);

  return (
    <div>
      <header
        style={{
          borderBottom: "1px solid var(--color-border)",
          background: "var(--color-surface)",
        }}
      >
        <div
          style={{
            maxWidth: "960px",
            margin: "0 auto",
            padding: "14px 24px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <span style={{ fontWeight: 600, fontSize: "15px" }}>DocuMind</span>

          <span
            style={{
              display: "flex",
              alignItems: "center",
              gap: "6px",
              fontSize: "12px",
              color: "var(--color-text-muted)",
            }}
          >
            <span
              style={{
                width: "7px",
                height: "7px",
                borderRadius: "999px",
                background: connected ? "var(--color-success-text)" : "var(--color-error-text)",
              }}
            />
            {connected === null ? "Checking…" : connected ? "Connected" : "Disconnected"}
          </span>
        </div>
      </header>

      {activeDocument ? (
        <ChatPage document={activeDocument} onBack={() => setActiveDocument(null)} />
      ) : (
        <DocumentsPage onOpenChat={setActiveDocument} />
      )}
    </div>
  );
}
