import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
from pathlib import Path
from scipy.signal import savgol_filter

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR  = ROOT / "data"
FFMPEG    = str(ROOT.parent / "ffmpeg" / "bin" / "ffmpeg.exe")
OUT_MP4   = ROOT / "norris_silverstone_2025_gforce.mp4"

matplotlib.rcParams["animation.ffmpeg_path"] = FFMPEG

# ── Load telemetry ─────────────────────────────────────────────────────────────
tel = pd.read_csv(DATA_DIR / "norris_2025_british_gp_fastest_lap.csv")
pos = pd.read_csv(DATA_DIR / "norris_2025_british_gp_fastest_lap_pos10hz.csv")
t        = tel["TimeSeconds"].values
x        = tel["X"].values
y        = tel["Y"].values
speed_ms = tel["Speed"].values / 3.6
speed_kh = tel["Speed"].values

# ── G-force calculation ────────────────────────────────────────────────────────
long_accel = np.gradient(speed_ms, t)
long_g     = long_accel / 9.81

# Lateral: heading-rate method on native 10Hz position data (see HANDOVER.md
# for why double-differentiating the linearly-interpolated 240Hz stream
# destroys curvature and underestimates fast-corner G).
t10 = pos["TimeSeconds"].values
x10 = pos["X"].values
y10 = pos["Y"].values

heading  = np.unwrap(np.arctan2(np.gradient(y10, t10), np.gradient(x10, t10)))
yaw_rate = np.gradient(heading, t10)  # rad/s

speed_ms_10  = np.interp(t10, t, speed_ms)
lat_accel_10 = np.abs(speed_ms_10 * yaw_rate)
lat_accel_10 = savgol_filter(lat_accel_10, window_length=5, polyorder=2)
lat_g_10     = lat_accel_10 / 9.81

lat_g = np.interp(t, t10, lat_g_10)  # back onto 240Hz grid for the rest of the pipeline

total_g = np.sqrt(long_g**2 + lat_g**2)
kernel  = np.ones(7) / 7
total_g = np.convolve(total_g, kernel, mode="same")
total_g = np.clip(total_g, 0, 6.5)

# ── Build segments + pre-compute RGBA ─────────────────────────────────────────
points   = np.stack([x, y], axis=1).reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
n_seg    = len(segments)

norm = mcolors.Normalize(vmin=0, vmax=5.5)
cmap = plt.cm.RdYlGn_r
g_vals = total_g[:-1]

rgba_full = cmap(norm(g_vals))  # (n_seg, 4)  — fully opaque target colors
rgba_init = rgba_full.copy()
rgba_init[:, 3] = 0.0           # start fully transparent

# ── Interpolate to 30 fps ─────────────────────────────────────────────────────
FPS       = 30
lap_time  = t[-1] - t[0]
n_frames  = int(np.ceil(lap_time * FPS)) + 1
t_uniform = np.linspace(t[0], t[-1], n_frames)

x_anim     = np.interp(t_uniform, t, x)
y_anim     = np.interp(t_uniform, t, y)
speed_anim = np.interp(t_uniform, t, speed_kh)
g_anim     = np.interp(t_uniform, t, total_g)

# frame → segment index (which segment the "car" is past)
seg_at = (t_uniform - t[0]) / (t[-1] - t[0]) * (n_seg - 1)
seg_at = seg_at.astype(int).clip(0, n_seg - 1)

# ── Figure setup ──────────────────────────────────────────────────────────────
BG   = "#0d0d0d"
fig, ax = plt.subplots(figsize=(16, 9), dpi=120, facecolor=BG)
ax.set_facecolor(BG)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Ghost track
ghost_colors = np.full((n_seg, 4), [1.0, 1.0, 1.0, 0.06])
lc_ghost = LineCollection(segments, colors=ghost_colors, linewidth=5, zorder=1)
ax.add_collection(lc_ghost)

# Shadow for the live trail
shadow_colors = rgba_init.copy()
shadow_colors[:, :3] = 0.0
shadow_colors[:, 3]  = 0.0
lc_shadow = LineCollection(segments, colors=shadow_colors, linewidth=9, zorder=2)
ax.add_collection(lc_shadow)

# Main G-force trail
live_colors = rgba_init.copy()
lc = LineCollection(segments, colors=live_colors, linewidth=5, zorder=3)
ax.add_collection(lc)

# Car dot — glow + core
glow, = ax.plot([], [], "o", color="#ffcc00", markersize=22, alpha=0.18, zorder=4)
dot,  = ax.plot([], [], "o", color="white",  markersize=7,  zorder=5)

ax.autoscale_view()
ax.set_aspect("equal")
ax.axis("off")

# NOTE: ax.set_aspect("equal") + autoscale_view() shrinks the Axes bounding
# box itself to fit the track's data aspect ratio (it ends up spanning only
# figure-x ~0.335-0.665, not the full width) — any HUD text placed with
# ax.transAxes therefore gets squeezed into that same narrow band as the
# track and can overlap it. All HUD text below uses a transparent full-figure
# overlay Axes instead (like the colorbar already does via fig.add_axes),
# positioned safely outside that 0.335-0.665 band. fig.text() was tried first
# but silently doesn't participate in FuncAnimation's blit machinery, so a
# real overlay Axes is required for anything updated per-frame.
overlay_ax = fig.add_axes([0, 0, 1, 1])
overlay_ax.set_facecolor("none")
overlay_ax.axis("off")
overlay_ax.set_xlim(0, 1)
overlay_ax.set_ylim(0, 1)
overlay_ax.set_zorder(10)

# ── Static text elements ───────────────────────────────────────────────────────
overlay_ax.text(0.04, 0.93, "LANDO NORRIS", color="white", fontsize=22,
                 fontweight="bold", va="top", fontfamily="monospace")
overlay_ax.text(0.04, 0.87, "2025 BRITISH GRAND PRIX · SILVERSTONE",
                 color="#aaaaaa", fontsize=10, va="top", fontfamily="monospace")
overlay_ax.text(0.04, 0.83, "FASTEST LAP · 1:29.734",
                 color="#ffcc00", fontsize=10, va="top", fontfamily="monospace")

# ── G-force real-world reference bands (upper-right annotation) ───────────────
# (max_g, icon, label, real-world example, color)
GFORCE_BANDS = [
    (1.5, "\U0001F697", "~1G",  "SUDDEN HARD BRAKING",         "#4CAF50"),
    (2.5, "✈",          "~2G",  "AIRLINER SHARP BANKED TURN",  "#FFD54F"),
    (3.5, "\U0001F680", "~3G",  "SPACE SHUTTLE LAUNCH",        "#FF9800"),
    (4.5, "\U0001F6F7", "~4G",  "OLYMPIC BOBSLED RUN",         "#FF5252"),
    (5.5, "\U0001F6E9", "~5G",  "FIGHTER JET TIGHT TURN",      "#F44336"),
    (float("inf"), "\U0001F4A5", ">5G", "SEVERE CAR CRASH IMPACT (50G+)", "#E040FB"),
]

def active_band_index(g):
    for i, (max_g, *_rest) in enumerate(GFORCE_BANDS):
        if g < max_g:
            return i
    return len(GFORCE_BANDS) - 1

DIM_COLOR = "#4a4a4a"
ROW_X_ICON, ROW_X_LABEL = 0.70, 0.745
ROW_Y_START, ROW_DY = 0.90, 0.11

band_icon_texts, band_tier_texts, band_example_texts = [], [], []
for i, (_max_g, icon, tier, example, color) in enumerate(GFORCE_BANDS):
    ry = ROW_Y_START - i * ROW_DY
    band_icon_texts.append(overlay_ax.text(ROW_X_ICON, ry, icon,
                                            color=DIM_COLOR, fontsize=15, va="top", ha="left",
                                            fontfamily="Segoe UI Emoji"))
    band_tier_texts.append(overlay_ax.text(ROW_X_LABEL, ry, tier,
                                            color=DIM_COLOR, fontsize=9, va="top", ha="left",
                                            fontfamily="monospace"))
    band_example_texts.append(overlay_ax.text(ROW_X_LABEL, ry - 0.035, example,
                                               color=DIM_COLOR, fontsize=8.5, va="top", ha="left",
                                               fontfamily="monospace"))

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar_ax = fig.add_axes([0.88, 0.12, 0.012, 0.35])
cbar_ax.set_facecolor(BG)
cb = fig.colorbar(sm, cax=cbar_ax)
cb.set_label("G-Force", color="#aaaaaa", fontsize=9, labelpad=8)
cb.ax.yaxis.set_tick_params(color="#aaaaaa", labelsize=8)
plt.setp(cb.ax.yaxis.get_ticklabels(), color="#aaaaaa")
cb.ax.set_facecolor(BG)

# Live readout text (updated each frame) — lower-right, clear of the track
# (track's fig-fraction bounding box tops out at x≈0.665) and clear of the
# colorbar (x 0.88-0.892, y 0.12-0.47).
spd_text = overlay_ax.text(0.70, 0.14, "",
                   color="white", fontsize=13, va="bottom", fontfamily="monospace")
g_text   = overlay_ax.text(0.70, 0.09, "",
                   color="white", fontsize=13, va="bottom", fontfamily="monospace")
time_text = overlay_ax.text(0.70, 0.04, "",
                    color="#888888", fontsize=9, va="bottom", fontfamily="monospace")

# ── Animation update ──────────────────────────────────────────────────────────
def update(frame):
    si = seg_at[frame]  # how many segments are "behind" the car

    # Reveal trail colors up to si
    c_live   = rgba_init.copy()
    c_shadow = rgba_init.copy()
    if si > 0:
        c_live[:si]      = rgba_full[:si]
        c_shadow[:si]    = rgba_full[:si]
        c_shadow[:si, :3] = 0.0
        c_shadow[:si, 3]  = rgba_full[:si, 3] * 0.5

    lc.set_colors(c_live)
    lc_shadow.set_colors(c_shadow)

    # Move car dot
    cx, cy = x_anim[frame], y_anim[frame]
    glow.set_data([cx], [cy])
    dot.set_data([cx], [cy])

    # Update HUD
    spd_text.set_text(f"SPEED   {speed_anim[frame]:5.0f} km/h")
    g_text.set_text(  f"G-FORCE {g_anim[frame]:5.2f} g")
    elapsed = t_uniform[frame] - t_uniform[0]
    mins    = int(elapsed // 60)
    secs    = elapsed % 60
    time_text.set_text(f"LAP TIME  {mins}:{secs:06.3f}")

    # Light up the G-force band matching the current reading, dim the rest
    active = active_band_index(g_anim[frame])
    for i, (_max_g, _icon, _tier, _example, color) in enumerate(GFORCE_BANDS):
        lit = i == active
        c = color if lit else DIM_COLOR
        band_icon_texts[i].set_color(c)
        band_icon_texts[i].set_fontsize(20 if lit else 15)
        band_tier_texts[i].set_color(c)
        band_tier_texts[i].set_fontsize(11 if lit else 9)
        band_tier_texts[i].set_fontweight("bold" if lit else "normal")
        band_example_texts[i].set_color(c)
        band_example_texts[i].set_fontweight("bold" if lit else "normal")

    return (lc, lc_shadow, glow, dot, spd_text, g_text, time_text,
            *band_icon_texts, *band_tier_texts, *band_example_texts)


print(f"Rendering {n_frames} frames at {FPS} fps ({lap_time:.1f}s lap)...")
ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=True)

writer = animation.FFMpegWriter(fps=FPS, bitrate=8000,
                                 extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
ani.save(str(OUT_MP4), writer=writer)
print(f"Saved -> {OUT_MP4}")
