import { useEffect, useRef } from "react";
import { insightsForLap, splitByDrivers } from "../playback.js";

const LABELS = { pace: "Pace", tyres: "Tyres", battle: "Battle", pit: "Pit strategy", movers: "Movers" };

export default function Insights({ race, insights, lap, finished }) {
  const shown = useRef(null);
  const entry = Array.isArray(insights) ? insightsForLap(insights, lap, finished) : null;
  const n = entry ? entry.lap : null;
  // fade only for normal playback (same or next lap); a seek swaps instantly
  const fade = shown.current !== null && n !== null && Math.abs(n - shown.current) <= 1;
  useEffect(() => { shown.current = n; }, [n]);

  if (insights === false) return <section className="insights"><div className="note">Insights unavailable</div></section>;
  if (!entry) return <section className="insights" />;

  const colors = Object.fromEntries(race.drivers.map((d) => [d.code, `#${d.color}`]));
  const codes = race.drivers.map((d) => d.code);
  const paint = (text) => splitByDrivers(text, codes).map((p, i) =>
    p.code ? <span key={i} className="drv" style={{ color: colors[p.code] }}>{p.text}</span> : p.text);

  return (
    <section className="insights" aria-label={`Insights after lap ${entry.lap}`}>
      {entry.insights.map((ins, i) => (
        <article key={`${entry.lap}-${i}`} className={fade ? "card fade" : "card"}>
          <span className="tag">{LABELS[ins.type] ?? ins.type}</span>
          <h3>{paint(ins.headline)}</h3>
          <p>{paint(ins.detail)}</p>
        </article>
      ))}
    </section>
  );
}
