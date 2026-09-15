const STYLES = {
  completed: { text: "var(--color-success-text)", bg: "var(--color-success-bg)", label: "Completed" },
  pending: { text: "var(--color-pending-text)", bg: "var(--color-pending-bg)", label: "Pending" },
  failed: { text: "var(--color-error-text)", bg: "var(--color-error-bg)", label: "Failed" },
};

export default function StatusBadge({ status }) {
  const style = STYLES[status] ?? { text: "var(--color-text-muted)", bg: "var(--color-bg)", label: status };

  return (
    <span
      style={{
        display: "inline-block",
        padding: "2px 8px",
        borderRadius: "999px",
        fontSize: "12px",
        fontWeight: 500,
        color: style.text,
        background: style.bg,
      }}
    >
      {style.label}
    </span>
  );
}
