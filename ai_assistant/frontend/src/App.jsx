import { useEffect, useRef, useState } from "react";
import { streamChat } from "./sse.js";
import ToolCard from "./components/ToolCard.jsx";
import Feedback from "./components/Feedback.jsx";

const sessionId = crypto.randomUUID();

export default function App() {
  const [models, setModels] = useState([]);
  const [model, setModel] = useState("");
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const bottom = useRef(null);

  useEffect(() => {
    fetch("/api/models").then((r) => r.json()).then((d) => { setModels(d.models); setModel(d.default); }).catch(() => {});
  }, []);
  useEffect(() => { bottom.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  const patchLast = (fn) => setMessages((ms) => ms.map((m, i) => (i === ms.length - 1 ? fn(m) : m)));
  const settleFirstRunning = (tools, name, patch) => {
    const i = tools.findIndex((t) => t.name === name && t.status === "running");
    return i < 0 ? tools : tools.map((t, j) => (j === i ? { ...t, ...patch } : t));
  };

  async function send(e) {
    e.preventDefault();
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    setBusy(true);
    setMessages((ms) => [...ms, { role: "user", text }, { role: "assistant", text: "", tools: [], done: false }]);
    try {
      for await (const ev of streamChat({ message: text, sessionId, model })) {
        if (ev.type === "text_delta") patchLast((m) => ({ ...m, text: m.text + ev.text }));
        else if (ev.type === "tool_call")
          patchLast((m) => ({ ...m, tools: [...m.tools, { name: ev.name, args: ev.args, status: "running" }] }));
        else if (ev.type === "tool_result")
          patchLast((m) => ({ ...m, tools: settleFirstRunning(m.tools, ev.name, { status: "ok", result: ev.result }) }));
        else if (ev.type === "tool_retry")
          patchLast((m) => ({ ...m, tools: settleFirstRunning(m.tools, ev.name, { status: "retry", message: ev.message }) }));
        else if (ev.type === "done")
          patchLast((m) => ({ ...m, text: ev.answer, done: true, traceId: ev.trace_id, unverified: ev.unverified, nudge: ev.nudge }));
        else if (ev.type === "error")
          patchLast((m) => ({ ...m, text: `Error: ${ev.message}`, done: true, error: true }));
      }
    } catch (err) {
      patchLast((m) => ({ ...m, text: `Error: ${err.message}`, done: true, error: true }));
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="app">
      <header>
        <h1>F1 Podium Assistant</h1>
        <label>
          Model{" "}
          <select value={model} onChange={(e) => setModel(e.target.value)} disabled={busy}>
            {models.map((m) => <option key={m}>{m}</option>)}
          </select>
        </label>
      </header>
      <section className="log" aria-live="polite">
        {messages.length === 0 && <p className="hint">Ask about the Sepang prediction, e.g. “What if Russell qualified on pole?”</p>}
        {messages.map((m, i) => (
          <div key={i} className={`msg msg-${m.role}`}>
            {m.role === "assistant" && m.tools.map((t, j) => <ToolCard key={j} tool={t} />)}
            {m.text && <p className={m.error ? "error" : ""}>{m.text}</p>}
            {m.role === "assistant" && !m.done && <span className="typing">…</span>}
            {m.unverified?.length > 0 && <p className="warn">Unverified numbers: {m.unverified.join(", ")}</p>}
            {m.role === "assistant" && m.done && !m.error && m.traceId && <Feedback traceId={m.traceId} nudge={m.nudge} />}
          </div>
        ))}
        <div ref={bottom} />
      </section>
      <form onSubmit={send} className="composer">
        <input value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask about the Sepang podium…" maxLength={2000} disabled={busy} />
        <button type="submit" disabled={busy || !input.trim()}>Send</button>
      </form>
    </main>
  );
}
