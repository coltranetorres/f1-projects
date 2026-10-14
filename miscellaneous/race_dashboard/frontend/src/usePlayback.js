import { useCallback, useEffect, useRef, useState } from "react";
import { getFrames } from "./api.js";
import { FrameBuffer, advancePlayhead, RetryTracker, WINDOW_S, windowStart, windowsToFetch } from "./playback.js";

const MAX_FRAME_DT_S = 0.25; // wall-clock cap per rAF frame (backgrounded tabs)

export function usePlayback(race) {
  const buf = useRef(new FrameBuffer());
  const inflight = useRef(new Set());
  const tracker = useRef(new RetryTracker());
  const retryTimer = useRef(null);
  const lastError = useRef("");
  const [t, setT] = useState(0);
  const [playing, setPlayingState] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [error, setError] = useState(null);
  const [, bump] = useState(0);
  const tRef = useRef(0);
  const ensureRef = useRef(null);
  const duration = race.duration;
  const lastT = Math.max(duration - 1 / race.hz, 0); // sample() has no window at t >= duration

  const syncError = useCallback(() => {
    setError(tracker.current.hasFailures() ? lastError.current : null);
  }, []);

  const scheduleRetry = useCallback(() => {
    clearTimeout(retryTimer.current);
    retryTimer.current = null;
    const delay = tracker.current.nextRetryDelay(performance.now());
    if (delay === null) return;
    retryTimer.current = setTimeout(() => ensureRef.current(tRef.current), delay);
  }, []);

  const ensure = useCallback((rawTime, { force = false } = {}) => {
    const time = Math.min(Math.max(rawTime, 0), duration);
    const now = performance.now();
    const needed = windowsToFetch(time, buf.current, duration);
    const target = windowStart(time);
    // Drop expired failures for windows nobody needs any more so they cannot pin the error state.
    if (tracker.current.prune(needed, now)) { syncError(); scheduleRetry(); }
    for (const start of needed) {
      if (inflight.current.has(start)) continue;
      if (!(force && start === target) && !tracker.current.shouldFetch(start, now)) continue;
      inflight.current.add(start);
      getFrames(start, start + WINDOW_S)
        .then((w) => { buf.current.add(w); tracker.current.succeed(start); bump((n) => n + 1); })
        .catch((e) => { lastError.current = e.message; tracker.current.fail(start, performance.now()); })
        .finally(() => { inflight.current.delete(start); syncError(); scheduleRetry(); });
    }
  }, [duration, syncError, scheduleRetry]);
  ensureRef.current = ensure;

  const seek = useCallback((time) => {
    const clamped = Math.min(Math.max(time, 0), duration);
    tRef.current = clamped;
    setT(clamped);
    ensure(clamped, { force: true });
  }, [duration, ensure]);

  const setPlaying = useCallback((on) => {
    if (on && tRef.current >= duration - 1e-6) seek(0);
    setPlayingState(on);
  }, [duration, seek]);

  useEffect(() => { ensure(0); }, [ensure]);
  useEffect(() => () => clearTimeout(retryTimer.current), []);

  useEffect(() => {
    if (!playing) return undefined;
    let raf;
    let last = performance.now();
    const loop = (now) => {
      const wallDt = (now - last) / 1000;
      last = now;
      ensure(tRef.current);
      if (buf.current.sample(Math.min(tRef.current, lastT))) {   // only advance on data we actually have
        tRef.current = advancePlayhead(tRef.current, wallDt, speed, duration, MAX_FRAME_DT_S);
        setT(tRef.current);
        if (tRef.current >= duration) setPlayingState(false);
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, duration, lastT, ensure]);

  const view = buf.current.sample(Math.min(t, lastT));
  const stalled = playing && (view === null || error !== null);
  return { view, t, playing, speed, loading: view === null && !error, error, stalled, setPlaying, setSpeed, seek };
}
