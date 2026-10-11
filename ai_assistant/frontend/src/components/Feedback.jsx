import { useState } from "react";

export default function Feedback({ traceId, nudge }) {
  const [value, setValue] = useState(null);
  const [reason, setReason] = useState("");
  const [state, setState] = useState("idle"); // idle | sending | sent | error

  async function submit() {
    setState("sending");
    try {
      const res = await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ trace_id: traceId, value, reason }),
      });
      setState(res.ok ? "sent" : "error");
    } catch {
      setState("error");
    }
  }

  if (state === "sent") return <div className="feedback-done">Thanks — feedback saved to the trace.</div>;
  return (
    <div className={`feedback ${nudge ? "feedback-nudge" : ""}`}>
      {nudge && <span className="feedback-ask">Was this answer right?</span>}
      <button type="button" aria-label="Thumbs up" aria-pressed={value === 1} onClick={() => setValue(1)}>👍</button>
      <button type="button" aria-label="Thumbs down" aria-pressed={value === 0} onClick={() => setValue(0)}>👎</button>
      {value !== null && (
        <div className="feedback-form">
          <textarea
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            placeholder={value === 0 ? "What went wrong? (encouraged)" : "Anything that worked well? (optional)"}
            rows={2}
          />
          <button type="button" onClick={submit} disabled={state === "sending"}>Send</button>
          {state === "error" && <span className="error">Could not save feedback.</span>}
        </div>
      )}
    </div>
  );
}
