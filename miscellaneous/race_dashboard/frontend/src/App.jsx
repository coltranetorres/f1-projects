import { useEffect, useState } from "react";
import { getRace } from "./api.js";
import { usePlayback } from "./usePlayback.js";
import TrackMap from "./components/TrackMap.jsx";
import Leaderboard from "./components/Leaderboard.jsx";
import RaceStatus from "./components/RaceStatus.jsx";
import Controls from "./components/Controls.jsx";

function Dashboard({ race }) {
  const p = usePlayback(race);
  return (
    <div className="layout">
      <aside>
        {p.view ? <Leaderboard race={race} cars={p.view.cars} /> : <div className="note">Loading…</div>}
      </aside>
      <main>
        {p.view ? <RaceStatus race={race} view={p.view} /> : null}
        <div className="map-wrap">
          {p.view ? <TrackMap race={race} cars={p.view.cars} /> : <div className="note">Loading race data…</div>}
          {p.error ? <div className="note warn overlay">Reconnecting… ({p.error})</div> : null}
        </div>
        <Controls duration={race.duration} t={p.t} playing={p.playing} speed={p.speed}
                  onPlay={p.setPlaying} onSpeed={p.setSpeed} onSeek={p.seek} />
      </main>
    </div>
  );
}

export default function App() {
  const [race, setRace] = useState(null);
  const [error, setError] = useState(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    setError(null);
    getRace().then(setRace).catch((e) => setError(e.message));
  }, [attempt]);

  if (error) return <div className="note warn">Could not load the race ({error}). <button onClick={() => setAttempt((n) => n + 1)}>Retry</button></div>;
  if (!race) return <div className="note">Loading…</div>;
  return <Dashboard race={race} />;
}
