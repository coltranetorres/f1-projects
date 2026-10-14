import { formatGap } from "../playback.js";

export default function Leaderboard({ race, cars }) {
  const colors = Object.fromEntries(race.drivers.map((d) => [d.code, `#${d.color}`]));
  return (
    <table className="leaderboard">
      <thead>
        <tr><th>Position</th><th>Driver</th><th>Time</th></tr>
      </thead>
      <tbody>
        {cars.map((c) => (
          <tr key={c.code} className={c.status === 2 ? "dnf" : c.status === 3 ? "pit" : ""}>
            <td>{c.pos}</td>
            <td><span className="swatch" style={{ background: colors[c.code] }} />{c.code}</td>
            <td>{formatGap(c)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
