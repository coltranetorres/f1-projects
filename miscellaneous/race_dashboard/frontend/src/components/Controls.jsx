export default function Controls({ duration, t, playing, speed, onPlay, onSpeed, onSeek }) {
  return (
    <div className="controls">
      <button onClick={() => onPlay(!playing)}>{playing ? "Pause" : "Play"}</button>
      {[1, 4, 16].map((s) => (
        <button key={s} className={s === speed ? "active" : ""} onClick={() => onSpeed(s)}>{s}x</button>
      ))}
      <input type="range" min={0} max={duration} step={1} value={t} onChange={(e) => onSeek(Number(e.target.value))} />
    </div>
  );
}
