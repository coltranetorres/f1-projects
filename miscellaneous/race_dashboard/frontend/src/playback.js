export const WINDOW_S = 60;

export const windowStart = (t) => Math.floor(t / WINDOW_S) * WINDOW_S;

// New playhead after one rAF frame: wall dt clamped to [0, maxFrameDtS] (negative/NaN -> 0), scaled by speed, result in [0, duration].
export function advancePlayhead(t, wallDtS, speed, duration, maxFrameDtS = 0.25) {
  const dt = wallDtS > 0 ? Math.min(wallDtS, maxFrameDtS) : 0;
  return Math.min(duration, Math.max(0, t + dt * speed));
}

export class FrameBuffer {
  constructor() {
    this.windows = new Map();
  }

  add(w) {
    this.windows.set(w.t0, w);
  }

  has(start) {
    return this.windows.has(start);
  }

  sample(t) {
    if (!(t > 0)) t = 0; // a tiny negative playhead must never miss window 0
    const w = this.windows.get(windowStart(t));
    if (!w || w.frames === 0) return null;
    const f = (t - w.t0) * w.hz;
    const i = Math.min(Math.max(Math.floor(f), 0), w.frames - 1);
    let frac = Math.min(Math.max(f - i, 0), 1);

    let next = null;
    if (i + 1 < w.frames) next = { w, j: i + 1 };
    else {
      const nw = this.windows.get(w.t0 + WINDOW_S);
      if (nw && nw.frames > 0) next = { w: nw, j: 0 };
    }
    if (!next) frac = 0;

    const cars = Object.entries(w.drivers).map(([code, d]) => {
      const nd = next ? next.w.drivers[code] : null;
      const lerp = (a, b) => (nd ? a + (b - a) * frac : a);
      return {
        code,
        x: lerp(d.x[i], nd?.x[next.j]),
        y: lerp(d.y[i], nd?.y[next.j]),
        pos: d.pos[i],
        gap: d.gap[i],
        down: d.down[i],
        status: d.status[i],
      };
    });
    cars.sort((a, b) => a.pos - b.pos);
    return { lap: w.lap[i], lapTime: w.lap_time[i], cars };
  }
}

export function windowsToFetch(t, buffer, duration, ahead = 2) {
  const out = [];
  const first = windowStart(Math.max(t, 0));
  for (let k = 0; k <= ahead; k++) {
    const s = first + k * WINDOW_S;
    if (s < duration && !buffer.has(s)) out.push(s);
  }
  return out;
}

export const RETRY_MS = 2000;

// Per-window failure bookkeeping so one failing window neither blocks another nor hides its error.
export class RetryTracker {
  constructor(retryMs = RETRY_MS) {
    this.retryMs = retryMs;
    this.retryAt = new Map();
  }

  fail(start, now) {
    this.retryAt.set(start, now + this.retryMs);
  }

  succeed(start) {
    this.retryAt.delete(start);
  }

  shouldFetch(start, now) {
    const at = this.retryAt.get(start);
    return at === undefined || now >= at;
  }

  // Drop due failures for windows not in `needed`; true if any were dropped.
  prune(needed, now) {
    let pruned = false;
    for (const start of [...this.retryAt.keys()]) {
      if (!needed.includes(start) && this.shouldFetch(start, now)) {
        this.succeed(start);
        pruned = true;
      }
    }
    return pruned;
  }

  hasFailures() {
    return this.retryAt.size > 0;
  }

  nextRetryDelay(now) {
    if (!this.retryAt.size) return null;
    return Math.max(Math.min(...this.retryAt.values()) - now, 0);
  }
}

export function formatLapTime(seconds) {
  const m = Math.floor(seconds / 60);
  const r = seconds - m * 60;
  return `${m}:${r.toFixed(3).padStart(6, "0")}`;
}

export function formatGap(car) {
  if (car.status === 3) return "PIT";
  if (car.status === 2) return "DNF";
  if (car.pos === 1) return "-";
  if (car.down > 0) return `+${car.down} Lap${car.down > 1 ? "s" : ""}`;
  return `+${car.gap.toFixed(3)}`;
}

// The set for lap N is built from laps <= N, so show it once the leader has completed lap N.
export function insightsForLap(insights, lap, finished) {
  if (!Array.isArray(insights) || insights.length === 0) return null;
  const n = Math.min(Math.max(finished ? lap : lap - 1, 1), insights.length);
  return insights.find((e) => e.lap === n) ?? null;
}

export function splitByDrivers(text, codes) {
  if (!codes.length) return [{ text }];
  const re = new RegExp(`\\b(${codes.join("|")})\\b`, "g");
  const out = [];
  let last = 0;
  for (const m of text.matchAll(re)) {
    if (m.index > last) out.push({ text: text.slice(last, m.index) });
    out.push({ text: m[0], code: m[0] });
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push({ text: text.slice(last) });
  return out;
}
