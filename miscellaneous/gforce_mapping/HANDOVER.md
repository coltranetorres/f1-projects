# G-Force Calculation Fix — Handover Document

## Problem

The current lateral G-force calculation in `animate.py` and `gforce_map.py` is significantly underestimated. Fast corners like Maggotts–Becketts–Chapel show ~1–2g on the map when the real figure is ~5g. The track trail stays green through the most demanding section of Silverstone, which is visually and factually wrong.

## Root Cause

`fastest.get_telemetry()` merges two data streams at different native sample rates:

| Stream | Source | Native Rate |
|---|---|---|
| `Speed`, `Throttle`, `Brake`, etc. | CAN bus (car data) | **240 Hz** |
| `X`, `Y` position | F1 timing feed (position data) | **10 Hz** |

FastF1 linearly interpolates `X, Y` up to 240Hz to produce a unified telemetry DataFrame. Taking `np.gradient()` twice on linearly-interpolated position data means the path looks like connected straight-line segments — curvature is essentially destroyed. The second derivative of a linear interpolation is zero between real samples, spiking only at the interpolation joints. This is why `lat_g` is near-zero through fast corners.

## Current Implementation (broken)

```python
# animate.py / gforce_map.py — lines 31–36
dx  = np.gradient(x, t)        # x is 240Hz but mostly fake (linearly interpolated)
dy  = np.gradient(y, t)
d2x = np.gradient(dx, t)       # second derivative amplifies interpolation noise
d2y = np.gradient(dy, t)
curvature = np.abs(dx * d2y - dy * d2x) / (dx**2 + dy**2 + 1e-9)**1.5
lat_g = (speed_ms**2 * curvature) / 9.81
```

## The Fix

### Approach: heading-rate method at native 10Hz

Instead of double-differentiating position, compute the car's **heading angle** from position and differentiate it once. Lateral acceleration is then:

```
lat_accel = speed × d(heading)/dt
```

One differentiation instead of two = far less noise amplification. Working at native 10Hz instead of fake 240Hz = no interpolation artifact.

### Step 1 — Re-fetch raw position data separately

Modify `fetch_data.py` to also save the raw `pos_data` at its native 10Hz, before it gets merged and upsampled.

```python
# fetch_data.py — add after getting fastest lap
session = fastf1.get_session(2025, "British Grand Prix", "R")
session.load(telemetry=True, weather=False, messages=False)

norris_laps = session.laps.pick_drivers("NOR")
fastest = norris_laps.pick_fastest()

# Existing — merged 240Hz telemetry (keep for speed/throttle/brake)
tel = fastest.get_telemetry()

# New — raw position data at native 10Hz
pos = fastest.get_pos_data()   # returns DataFrame with Time, X, Y, Z at ~10Hz
pos["TimeSeconds"] = pos["Time"].dt.total_seconds()
pos = pos.drop(columns=["Time"])
pos.to_csv(DATA_DIR / "norris_2025_british_gp_fastest_lap_pos10hz.csv", index=False)
```

### Step 2 — Rewrite the lateral G calculation

In `animate.py` and `gforce_map.py`, replace the curvature block with the heading-rate method operating on the native 10Hz position data, then interpolate the result back onto the 240Hz time grid for the animation.

```python
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# Load native 10Hz position data
pos = pd.read_csv(DATA_DIR / "norris_2025_british_gp_fastest_lap_pos10hz.csv")
t10   = pos["TimeSeconds"].values
x10   = pos["X"].values
y10   = pos["Y"].values

# Compute heading angle at 10Hz
heading = np.arctan2(np.gradient(y10, t10), np.gradient(x10, t10))

# Unwrap heading to avoid ±π discontinuities on direction changes
heading = np.unwrap(heading)

# Heading rate (yaw rate) — one differentiation only
yaw_rate = np.gradient(heading, t10)   # rad/s

# Speed at 10Hz — interpolate from 240Hz speed channel
speed_ms_240 = tel["Speed"].values / 3.6
t240 = tel["TimeSeconds"].values
speed_ms_10 = np.interp(t10, t240, speed_ms_240)

# Lateral acceleration
lat_accel_10 = np.abs(speed_ms_10 * yaw_rate)

# Optional: smooth with Savitzky-Golay before interpolating back
# (preserves peak shape better than a box filter)
lat_accel_10 = savgol_filter(lat_accel_10, window_length=5, polyorder=2)

lat_g_10 = lat_accel_10 / 9.81

# Interpolate back to 240Hz time grid for the rest of the pipeline
lat_g = np.interp(t240, t10, lat_g_10)

# Longitudinal G — unchanged, already accurate
long_accel = np.gradient(speed_ms_240, t240)
long_g = long_accel / 9.81

# Total G — same as before
total_g = np.sqrt(long_g**2 + lat_g**2)
```

> **Note:** `scipy` is not in `pyproject.toml`. Add it with `uv add scipy`, or replace `savgol_filter` with a plain `np.convolve` box filter if you want to avoid the dependency.

### Step 3 — No changes needed to the animation logic

Everything in `animate.py` from the segment-building step onward (`points`, `segments`, `LineCollection`, frame rendering) stays exactly the same. Only the G-force calculation block changes.

## Expected Result After Fix

| Section | Before (broken) | After (fix) |
|---|---|---|
| Maggotts–Becketts lateral G | ~1–2g (green) | ~3–4g (orange/red) |
| Braking into Village | 3.16g (accurate, unchanged) | 3.16g (unchanged) |
| Overall peak G | 3.16g braking-dominant | Higher, corner-dominant |
| Track map color at fast corners | Green (wrong) | Orange–red (correct) |

The fix will not reach the true ~5g peak because 10Hz position data physically cannot resolve sub-second direction changes at full fidelity. But the result will be materially more accurate and visually correct — fast corners will light up as the highest-G sections, which is the ground truth.

## Files to Modify

| File | Change |
|---|---|
| `fetch_data.py` | Add `get_pos_data()` save alongside existing telemetry save |
| `animate.py` | Replace lateral G block (lines 31–36) with heading-rate method |
| `gforce_map.py` | Same replacement as `animate.py` |
| `pyproject.toml` | `uv add scipy` if using `savgol_filter` |

## Validation Check

After re-running, print the peak lateral G at the Maggotts–Becketts complex (roughly 30–45s into the lap at Silverstone). If the fix is working, it should read **3g+**. If it still reads under 2g, the `get_pos_data()` call may be returning the merged/upsampled stream — verify with `print(len(pos))`: native 10Hz for a ~90s lap should give ~900 rows, not ~21,000+.
