import { useState } from "react";

export default function ToolCard({ tool }) {
  const [open, setOpen] = useState(false);
  const label = { running: "running…", ok: "done", retry: "rejected, retrying" }[tool.status];
  return (
    <div className={`tool tool-${tool.status}`}>
      <button type="button" className="tool-head" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className="tool-name">{tool.name}</span>
        <span className="tool-args">{JSON.stringify(tool.args)}</span>
        <span className="tool-status">{label}</span>
      </button>
      {open && (
        <pre className="tool-body">
          {tool.status === "retry" ? tool.message : JSON.stringify(tool.result, null, 2)}
        </pre>
      )}
    </div>
  );
}
