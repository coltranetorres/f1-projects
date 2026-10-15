// playback.test.js
import { describe, expect, it } from "vitest";
import { FrameBuffer, advancePlayhead, RETRY_MS, RetryTracker, WINDOW_S, formatGap, formatLapTime, windowStart, windowsToFetch, insightsForLap, splitByDrivers } from "./playback.js";

const win = (t0, frames, over = {}) => ({
  t0, hz: 4, frames,
  lap: Array(frames).fill(3), lap_time: Array.from({ length: frames }, (_, i) => 80 + i / 4),
  drivers: {
    VER: { x: Array.from({ length: frames }, (_, i) => i * 4), y: Array(frames).fill(0), pos: Array(frames).fill(1), gap: Array(frames).fill(0), down: Array(frames).fill(0), status: Array(frames).fill(0) },
    NOR: { x: Array(frames).fill(0), y: Array(frames).fill(0), pos: Array(frames).fill(2), gap: Array(frames).fill(8.277), down: Array(frames).fill(0), status: Array(frames).fill(0) },
  },
  ...over,
});

describe("windowStart", () => {
  it("aligns to 60 s", () => {
    expect(windowStart(0)).toBe(0);
    expect(windowStart(59.9)).toBe(0);
    expect(windowStart(60)).toBe(60);
    expect(windowStart(125)).toBe(120);
  });
});

describe("FrameBuffer.sample", () => {
  it("returns null when the window is not loaded (seek into the unknown)", () => {
    expect(new FrameBuffer().sample(10)).toBeNull();
  });

  it("clamps a tiny negative playhead to t=0 instead of returning null", () => {
    const b = new FrameBuffer();
    b.add(win(0, 240));
    const s = b.sample(-0.002);
    expect(s).not.toBeNull();
    expect(s.cars[0].x).toBe(b.sample(0).cars[0].x);
  });

  it("returns null for an empty window", () => {
    const b = new FrameBuffer();
    b.add(win(600, 0));
    expect(b.sample(610)).toBeNull();
  });

  it("interpolates x between frames and sorts cars by position", () => {
    const b = new FrameBuffer();
    b.add(win(0, 240));
    const s = b.sample(0.125); // halfway between frame 0 (x=0) and frame 1 (x=4)
    expect(s.cars.map((c) => c.code)).toEqual(["VER", "NOR"]);
    expect(s.cars[0].x).toBeCloseTo(2);
    expect(s.lap).toBe(3);
  });

  it("interpolates across a window boundary when the next window is loaded", () => {
    const b = new FrameBuffer();
    b.add(win(0, 240));
    const next = win(60, 240);
    next.drivers.VER.x[0] = 1000;
    b.add(next);
    const last = b.sample(59.875); // halfway between frame 239 (x=956) and next window frame 0 (x=1000)
    expect(last.cars[0].x).toBeCloseTo(978);
  });

  it("holds the last frame at a window end when the next window is missing", () => {
    const b = new FrameBuffer();
    b.add(win(0, 240));
    expect(b.sample(59.875).cars[0].x).toBe(956);
  });
});

describe("windowsToFetch", () => {
  it("lists unloaded windows from the playhead through the lookahead, within the race", () => {
    const b = new FrameBuffer();
    b.add(win(0, 240));
    expect(windowsToFetch(10, b, 1000, 2)).toEqual([60, 120]);
    expect(windowsToFetch(10, b, 100, 2)).toEqual([60]);
  });

  it("never requests a negative window start", () => {
    expect(windowsToFetch(-0.002, new FrameBuffer(), 1000)).toEqual([0, 60, 120]);
  });
});

describe("advancePlayhead", () => {
  it("never moves backwards on a negative wall dt and never goes below 0", () => {
    expect(advancePlayhead(0, -0.002, 1, 1000)).toBe(0);
    expect(advancePlayhead(5, -1, 4, 1000)).toBe(5);
  });
  it("treats NaN dt as 0", () => {
    expect(advancePlayhead(5, NaN, 1, 1000)).toBe(5);
  });
  it("caps the wall dt", () => {
    expect(advancePlayhead(10, 5, 1, 1000)).toBeCloseTo(10.25);
    expect(advancePlayhead(10, 5, 1, 1000, 0.5)).toBeCloseTo(10.5);
  });
  it("multiplies by speed", () => {
    expect(advancePlayhead(10, 0.1, 4, 1000)).toBeCloseTo(10.4);
  });
  it("clamps at duration", () => {
    expect(advancePlayhead(999.9, 0.2, 4, 1000)).toBe(1000);
  });
  it("keeps t >= 0 even for a negative start", () => {
    expect(advancePlayhead(-0.5, 0, 1, 1000)).toBe(0);
  });
});

describe("RetryTracker", () => {
  it("respects backoff per window without blocking others", () => {
    const r = new RetryTracker();
    expect(r.shouldFetch(0, 0)).toBe(true);
    r.fail(0, 1000);
    expect(r.shouldFetch(0, 1000 + RETRY_MS - 1)).toBe(false);
    expect(r.shouldFetch(0, 1000 + RETRY_MS)).toBe(true);
    expect(r.shouldFetch(60, 1000)).toBe(true);
  });
  it("succeed clears and hasFailures tracks remaining windows", () => {
    const r = new RetryTracker();
    expect(r.hasFailures()).toBe(false);
    r.fail(0, 0);
    r.fail(60, 0);
    r.succeed(0);
    expect(r.hasFailures()).toBe(true);
    expect(r.shouldFetch(0, 0)).toBe(true);
    r.succeed(60);
    expect(r.hasFailures()).toBe(false);
  });
  it("nextRetryDelay is time to earliest retry", () => {
    const r = new RetryTracker();
    expect(r.nextRetryDelay(0)).toBeNull();
    r.fail(0, 100);
    r.fail(60, 500);
    expect(r.nextRetryDelay(200)).toBe(RETRY_MS - 100);
    expect(r.nextRetryDelay(10000)).toBe(0);
  });
});

describe("RetryTracker.prune", () => {
  it("drops only due failures for windows not needed and reports it", () => {
    const r = new RetryTracker();
    r.fail(0, 0);
    r.fail(60, 0);
    r.fail(120, 5000);
    expect(r.prune([60], RETRY_MS)).toBe(true);
    expect([...r.retryAt.keys()].sort((a, b) => a - b)).toEqual([60, 120]);
    expect(r.prune([60], RETRY_MS)).toBe(false);
  });
});

describe("formatters", () => {
  it("formats lap times", () => {
    expect(formatLapTime(81.852)).toBe("1:21.852");
    expect(formatLapTime(65.5)).toBe("1:05.500");
  });
  it("formats gaps", () => {
    expect(formatGap({ pos: 1, status: 0, gap: 0, down: 0 })).toBe("-");
    expect(formatGap({ pos: 2, status: 0, gap: 8.277, down: 0 })).toBe("+8.277");
    expect(formatGap({ pos: 14, status: 0, gap: 90, down: 1 })).toBe("+1 Lap");
    expect(formatGap({ pos: 15, status: 0, gap: 90, down: 2 })).toBe("+2 Laps");
    expect(formatGap({ pos: 20, status: 2, gap: 0, down: 0 })).toBe("DNF");
  });

  it("shows PIT for pit-lane starters, before any P1 / gap logic", () => {
    expect(formatGap({ pos: 22, status: 3, gap: 0, down: 0 })).toBe("PIT");
    expect(formatGap({ pos: 1, status: 3, gap: 0, down: 0 })).toBe("PIT");
    expect(formatGap({ pos: 21, status: 3, gap: 12, down: 1 })).toBe("PIT");
  });
});


describe("insightsForLap", () => {
  const sets = Array.from({ length: 10 }, (_, i) => ({ lap: i + 1, insights: [] }));
  it("shows the previous completed lap while the leader is mid-lap", () => {
    expect(insightsForLap(sets, 5, false).lap).toBe(4);
  });
  it("shows the lap-1 set while the leader is on lap 1", () => {
    expect(insightsForLap(sets, 1, false).lap).toBe(1);
  });
  it("shows the last lap's set once the leader has finished", () => {
    expect(insightsForLap(sets, 10, true).lap).toBe(10);
    expect(insightsForLap(sets, 10, false).lap).toBe(9);
  });
  it("clamps to the cached range and tolerates empty data", () => {
    expect(insightsForLap(sets, 99, false).lap).toBe(10);
    expect(insightsForLap([], 3, false)).toBeNull();
    expect(insightsForLap(undefined, 3, false)).toBeNull();
  });
});

describe("splitByDrivers", () => {
  it("splits driver codes out of text", () => {
    expect(splitByDrivers("NOR chases VER, not NORRIS", ["NOR", "VER"])).toEqual([
      { text: "NOR", code: "NOR" }, { text: " chases " }, { text: "VER", code: "VER" }, { text: ", not NORRIS" },
    ]);
  });
  it("returns the text unchanged with no codes", () => {
    expect(splitByDrivers("hello", [])).toEqual([{ text: "hello" }]);
  });
});
