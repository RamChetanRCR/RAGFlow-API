import { useState, useRef, useEffect, FormEvent } from "react";

type Message = {
  role: "user" | "assistant";
  content: string;
};

export default function Home() {
  const [query, setQuery] = useState("");
  const [messages, setMessages] = useState<Message[]>([
    { role: "assistant", content: "Ask me anything about your documents." },
  ]);
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!query.trim() || loading) return;

    const userMsg: Message = { role: "user", content: query };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, stream: false }),
      });
      const data = await res.json();
      const assistantMsg: Message = {
        role: "assistant",
        content: data.answer || "No answer generated.",
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Error connecting to server." },
      ]);
    }

    setQuery("");
    setLoading(false);
  };

  const handleUpload = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const form = e.currentTarget;
    const fileInput = form.querySelector<HTMLInputElement>('input[type="file"]');
    if (!fileInput?.files?.length) return;

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    try {
      const res = await fetch("/api/ingest", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Ingested: ${data.filename} (${data.page_count} pages, ${data.chunk_count} chunks) — doc_id: ${data.doc_id}`,
        },
      ]);
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Upload failed." },
      ]);
    }

    fileInput.value = "";
  };

  return (
    <div style={{ maxWidth: 800, margin: "0 auto", padding: 20 }}>
      <h1 style={{ borderBottom: "2px solid #333", paddingBottom: 10 }}>
        RAGFlow Chat
      </h1>

      <form onSubmit={handleUpload} style={{ marginBottom: 16 }}>
        <input type="file" accept=".pdf" style={{ marginRight: 8 }} />
        <button type="submit" style={{ padding: "6px 16px" }}>
          Upload PDF
        </button>
      </form>

      <div
        style={{
          border: "1px solid #ccc",
          borderRadius: 8,
          padding: 16,
          minHeight: 400,
          maxHeight: 500,
          overflowY: "auto",
          marginBottom: 16,
          background: "#fafafa",
        }}
      >
        {messages.map((msg, i) => (
          <div
            key={i}
            style={{
              marginBottom: 12,
              textAlign: msg.role === "user" ? "right" : "left",
            }}
          >
            <span
              style={{
                display: "inline-block",
                padding: "8px 14px",
                borderRadius: 12,
                background: msg.role === "user" ? "#0070f3" : "#e5e5ea",
                color: msg.role === "user" ? "#fff" : "#000",
                maxWidth: "80%",
                whiteSpace: "pre-wrap",
              }}
            >
              {msg.content}
            </span>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <form onSubmit={handleSubmit} style={{ display: "flex", gap: 8 }}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question about your documents..."
          disabled={loading}
          style={{
            flex: 1,
            padding: 10,
            fontSize: 16,
            borderRadius: 6,
            border: "1px solid #ccc",
          }}
        />
        <button
          type="submit"
          disabled={loading}
          style={{
            padding: "10px 20px",
            fontSize: 16,
            background: "#0070f3",
            color: "#fff",
            border: "none",
            borderRadius: 6,
            cursor: loading ? "not-allowed" : "pointer",
          }}
        >
          {loading ? "..." : "Send"}
        </button>
      </form>
    </div>
  );
}
