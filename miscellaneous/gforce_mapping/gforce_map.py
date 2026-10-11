import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from pathlib import Path
from scipy.signal import savgol_filter

DATA_DIR = Path(__file__).parent / "data"
OUT_PNG = Path(__file__).parent / "norris_silverstone_2025_gforce_map.png"

tel = pd.read_csv(DATA_DIR / "norris_2025_british_gp_fastest_lap.csv")
pos = pd.read_csv(DATA_DIR / "norris_2025_british_gp_fastest_lap_pos10hz.csv")

t = tel["TimeSeconds"].values
x = tel["X"].values
y = tel["Y"].values
speed_ms = tel["Speed"].values / 3.6  # km/h -> m/s

# ── G-force calculation ────────────────────────────────────────────────────────
# Longitudinal: rate of speed change (accurate at native 240Hz CAN data)
long_accel = np.gradient(speed_ms, t)
long_g = long_accel / 9.81

# Lateral: heading-rate method on native 10Hz position data.
# Double-differentiating linearly-interpolated 240Hz position destroys
# curvature; instead differentiate heading once at 10Hz.
t10 = pos["TimeSeconds"].values
x10 = pos["X"].values
y10 = pos["Y"].values

heading = np.unwrap(np.arctan2(np.gradient(y10, t10), np.gradient(x10, t10)))
yaw_rate = np.gradient(heading, t10)  # rad/s

speed_ms_10 = np.interp(t10, t, speed_ms)
lat_accel_10 = np.abs(speed_ms_10 * yaw_rate)
lat_accel_10 = savgol_filter(lat_accel_10, window_length=5, polyorder=2)
lat_g_10 = lat_accel_10 / 9.81

# Interpolate back onto the 240Hz time grid used by the rest of the pipeline
lat_g = np.interp(t, t10, lat_g_10)

total_g = np.sqrt(long_g**2 + lat_g**2)

# Smooth with a simple box filter (≈5-sample window ~0.1 s)
kernel = np.ones(5) / 5
total_g_s = np.convolve(total_g, kernel, mode="same")
total_g_s = np.clip(total_g_s, 0, 6.5)

# ── Track map segments ─────────────────────────────────────────────────────────
points   = np.stack([x, y], axis=1).reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
g_vals   = total_g_s[:-1]

# ── Plot ───────────────────────────────────────────────────────────────────────
BG = "#0d0d0d"
fig, ax = plt.subplots(figsize=(13, 11), facecolor=BG)
ax.set_facecolor(BG)

norm = mcolors.Normalize(vmin=0, vmax=5.5)
cmap = plt.cm.RdYlGn_r   # green = coasting, red = peak G

# Shadow layer for depth
lc_shadow = LineCollection(segments, linewidth=8, color="#000000", alpha=0.5, zorder=1)
ax.add_collection(lc_shadow)

lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=4.5, zorder=2, capstyle="round")
lc.set_array(g_vals)
ax.add_collection(lc)

# Start/finish marker
ax.scatter(x[0], y[0], s=120, color="white", zorder=5, linewidths=1.5, edgecolors="#333")
ax.annotate("S/F", (x[0], y[0]), color="white", fontsize=8, ha="left",
            xytext=(80, 30), textcoords="offset points",
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.8))

# Colorbar
cbar = plt.colorbar(lc, ax=ax, fraction=0.025, pad=0.02, aspect=30)
cbar.set_label("Total G-Force (g)", color="white", fontsize=11, labelpad=10)
cbar.ax.yaxis.set_tick_params(color="white", labelsize=9)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
cbar.ax.set_facecolor(BG)

# Labels
peak = total_g_s.max()
ax.set_title(
    "Lando Norris  ·  2025 British Grand Prix — Silverstone\nFastest Lap G-Force Map",
    color="white", fontsize=14, fontweight="bold", pad=14
)
ax.text(
    0.01, 0.02,
    f"Peak: {peak:.2f} g  ·  Maggotts–Becketts–Chapel sequence visible top-left",
    transform=ax.transAxes, color="#aaaaaa", fontsize=8.5
)

ax.autoscale_view()
ax.set_aspect("equal")
ax.axis("off")
plt.tight_layout()

fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight", facecolor=BG)
print(f"Saved -> {OUT_PNG}")
plt.show()
