from dataclasses import dataclass
from statistics import median

TYPES = ("pace", "tyres", "battle", "pit", "movers")
PACE_WINDOW = 3
PACE_LOOKBACK = 6


@dataclass(frozen=True)
class LapRecord:
    driver: str
    lap: int
    time_s: float | None
    end_s: float | None       # session time (s) at the end of the lap
    position: int | None      # running position at the end of the lap
    compound: str | None
    tyre_life: int | None
    stint: int | None
    pit_in: bool
    pit_out: bool
    clean: bool               # accurate, green flag, not an in/out lap, not lap 1


@dataclass(frozen=True)
class Candidate:
    type: str
    key: str
    drivers: tuple[str, ...]
    numbers: dict
    score: float              # heuristic notability, 0..1

    def to_dict(self) -> dict:
        return {"type": self.type, "key": self.key, "drivers": list(self.drivers),
                "numbers": self.numbers, "score": round(self.score, 3)}


class LapTable:
    def __init__(self, records, total_laps: int, grid: dict[str, int]):
        self.total_laps = total_laps
        self.grid = dict(grid)
        self._rec = {(r.driver, r.lap): r for r in records}
        self.drivers = sorted({r.driver for r in records})
        self.max_lap = max((r.lap for r in records), default=0)

    def rec(self, driver, lap):
        return self._rec.get((driver, lap))

    def position(self, driver, lap):
        if lap <= 0:
            return self.grid.get(driver)
        r = self.rec(driver, lap)
        return r.position if r else None

    def order(self, lap) -> list[str]:
        ranked = [(r.position, d) for d in self.drivers
                  if (r := self.rec(d, lap)) is not None and r.position is not None]
        return [d for _, d in sorted(ranked)]

    def gap_ahead(self, driver, lap):
        order = self.order(lap)
        if driver not in order or order.index(driver) == 0:
            return None
        a, b = self.rec(order[order.index(driver) - 1], lap), self.rec(driver, lap)
        if a is None or b is None or a.end_s is None or b.end_s is None:
            return None
        gap = b.end_s - a.end_s
        return gap if gap >= 0 else None

    def recent_pace(self, driver, lap, n=PACE_WINDOW):
        times = []
        for l in range(lap, max(lap - PACE_LOOKBACK, 0), -1):
            r = self.rec(driver, l)
            if r is not None and r.clean and r.time_s:
                times.append(r.time_s)
                if len(times) == n:
                    break
        return sum(times) / len(times) if len(times) >= 2 else None

    def tyre(self, driver, lap):
        r = self.rec(driver, lap)
        if r is None or r.compound is None or r.tyre_life is None:
            return None
        return r.compound, r.tyre_life

    def stint_laps(self, driver, lap) -> list[LapRecord]:
        cur = self.rec(driver, lap)
        if cur is None or cur.stint is None:
            return []
        out = []
        for l in range(lap, 0, -1):
            r = self.rec(driver, l)
            if r is None or r.stint != cur.stint:
                break
            if r.clean and r.time_s and r.tyre_life is not None:
                out.append(r)
        return list(reversed(out))

    def pit_loss(self, lap):
        """Median extra time of completed stops (in-lap + out-lap minus two clean laps), or None."""
        losses = []
        for d in self.drivers:
            ref = [r.time_s for l in range(1, lap + 1)
                   if (r := self.rec(d, l)) is not None and r.clean and r.time_s]
            if len(ref) < 3:
                continue
            base = median(ref)
            for l in range(1, lap):
                a, b = self.rec(d, l), self.rec(d, l + 1)
                if a and b and a.pit_in and b.pit_out and a.time_s and b.time_s:
                    losses.append(a.time_s + b.time_s - 2 * base)
        return median(losses) if losses else None
