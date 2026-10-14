import { formatLapTime } from "../playback.js";

export default function RaceStatus({ race, view }) {
  const best = race.best_lap;
  return (
    <section className="status">
      <h2>Race status · {race.event}</h2>
      <div className="status-grid">
        <div><span className="label">Lap</span><span className="big">{view.lap}<small>/{race.total_laps}</small></span></div>
        <div><span className="label">Current lap</span><span className="mid">{formatLapTime(view.lapTime)}</span></div>
        <div><span className="label">Best lap ({best.driver})</span><span className="mid">{formatLapTime(best.time_s)}</span></div>
      </div>
    </section>
  );
}
