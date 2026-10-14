import { useMemo } from "react";

export default function TrackMap({ race, cars }) {
  const { path, box, r, flipY } = useMemo(() => {
    const xs = race.outline.map((p) => p[0]);
    const ys = race.outline.map((p) => p[1]);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);
    const pad = (maxX - minX) * 0.06;
    const flipY = (y) => maxY + minY - y; // SVG y grows downward
    return {
      path: race.outline.map((p, i) => `${i ? "L" : "M"}${p[0]},${flipY(p[1])}`).join(" ") + "Z",
      box: `${minX - pad} ${minY - pad} ${maxX - minX + 2 * pad} ${maxY - minY + 2 * pad}`,
      r: (maxX - minX) * 0.011,
      flipY,
    };
  }, [race]);

  const colors = useMemo(() => Object.fromEntries(race.drivers.map((d) => [d.code, `#${d.color}`])), [race]);

  return (
    <svg className="trackmap" viewBox={box} preserveAspectRatio="xMidYMid meet">
      <path d={path} className="track-outline" strokeWidth={r * 1.6} />
      {cars.map((c) => {
        const cy = flipY(c.y);
        const above = c.pos % 2 === 0; // alternate label side so neighbours in a pack don't collide
        return (
          <g key={c.code} opacity={c.status === 2 ? 0.35 : c.status === 3 ? 0.45 : 1}>
            <circle cx={c.x} cy={cy} r={r} fill={colors[c.code]} className="dot" strokeWidth={r * 0.25} />
            <text x={c.x + r * 1.4} y={cy + (above ? -r * 1.2 : r * 2.2)} fontSize={r * 2.4} className="dot-label">
              {c.status === 3 ? `${c.code}·PIT` : c.code}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
