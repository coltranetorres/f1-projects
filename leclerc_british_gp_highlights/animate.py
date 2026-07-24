import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
FFMPEG = str(ROOT.parent / "ffmpeg" / "bin" / "ffmpeg.exe")
OUT_MP4 = ROOT / "leclerc_british_gp_highlights.mp4"

matplotlib.rcParams["animation.ffmpeg_path"] = FFMPEG

FPS = 30
BG = "#0d0d0d"
COL_LEC = "#ED1131"   # Ferrari
COL_HAM = "#FF7FA3"   # Ferrari teammate, lighter to distinguish from LEC
COL_ANT = "#00D7B6"   # Mercedes
DIM = "#4a4a4a"
SPEED_MAX = 340.0     # km/h, bar scale headroom over the ~320 km/h observed peak

# ─────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────
h1_lec = pd.read_csv(DATA_DIR / "lap1_start_LEC_telemetry.csv")
h1_ant = pd.read_csv(DATA_DIR / "lap1_start_ANT_telemetry.csv")

h2_lec = pd.read_csv(DATA_DIR / "laps36_48_LEC_telemetry.csv")
h2_ant = pd.read_csv(DATA_DIR / "laps36_48_ANT_telemetry.csv")
h2_ham = pd.read_csv(DATA_DIR / "laps36_48_HAM_telemetry.csv")
gap_timeline = pd.read_csv(DATA_DIR / "laps36_48_gap_timeline.csv")
race_laps = pd.read_csv(DATA_DIR / "R09_British_GP_laps.csv")

h3_lec = pd.read_csv(DATA_DIR / "leclerc_fastest_lap_telemetry.csv")

corners = pd.read_csv(DATA_DIR / "silverstone_corners_named.csv")
SIGNATURE_CORNERS = ["Copse", "Maggotts", "Becketts", "Stowe"]

# Full-lap outline for the ghost track (fastest lap covers one complete loop)
ghost_x = h3_lec["X"].values
ghost_y = h3_lec["Y"].values


def dist_to_xy(dist_value, ref_df=h3_lec):
    x = np.interp(dist_value, ref_df["Distance"].values, ref_df["X"].values)
    y = np.interp(dist_value, ref_df["Distance"].values, ref_df["Y"].values)
    return x, y


def interp_driver(df, t_col, t_grid):
    return {
        "x": np.interp(t_grid, df[t_col], df["X"]),
        "y": np.interp(t_grid, df[t_col], df["Y"]),
        "speed": np.interp(t_grid, df[t_col], df["Speed"]),
        "brake": np.interp(t_grid, df[t_col], df["Brake"].astype(float)),
        "dist": np.interp(t_grid, df[t_col], df["Distance"]),
    }


# ─────────────────────────────────────────────────────────────────────────
# Segment A — Lap 1 launch, LEC vs ANT, lights-out to Turn 1 (Abbey)
# ─────────────────────────────────────────────────────────────────────────
A_DURATION = h1_lec["TimeSeconds"].max()
A_FRAMES = int(A_DURATION * FPS)
a_t = np.linspace(0, A_DURATION, A_FRAMES)

a_lec = interp_driver(h1_lec, "TimeSeconds", a_t)
a_ant = interp_driver(h1_ant, "TimeSeconds", a_t)

# ─────────────────────────────────────────────────────────────────────────
# Segment B — Lap 41-42 only: the moment Antonelli collapses, identified
# from race data below. Lap 41 is his in-lap (issue develops, pits at the
# end of it); Lap 42 is the lap that actually costs him track position —
# 2:17.8 vs a ~1:32 normal lap, and Position drops from P2 to P6 that lap.
# ─────────────────────────────────────────────────────────────────────────
ant_laps = race_laps[race_laps["Driver"] == "ANT"].copy()
ant_laps["LapStartSeconds"] = pd.to_timedelta(ant_laps["LapStartTime"]).dt.total_seconds()
ant_laps["TimeSeconds"] = pd.to_timedelta(ant_laps["Time"]).dt.total_seconds()

collapse_start = ant_laps.loc[ant_laps["LapNumber"] == 41, "LapStartSeconds"].iloc[0]
collapse_end = ant_laps.loc[ant_laps["LapNumber"] == 42, "TimeSeconds"].iloc[0]

B_SECONDS = 16
B_FRAMES = B_SECONDS * FPS
b_t = np.linspace(collapse_start, collapse_end, B_FRAMES)

b_lec = interp_driver(h2_lec, "SessionSeconds", b_t)
b_ant = interp_driver(h2_ant, "SessionSeconds", b_t)
b_ham = interp_driver(h2_ham, "SessionSeconds", b_t)
b_lap = np.interp(b_t, h2_lec["SessionSeconds"], h2_lec["LapNumber"])


def gap_lookup(driver):
    sub = gap_timeline[gap_timeline["Driver"] == driver].sort_values("TimeSeconds")
    return sub["TimeSeconds"].values, sub["GapToLeaderSeconds"].values


ant_gap_t, ant_gap_v = gap_lookup("ANT")
ham_gap_t, ham_gap_v = gap_lookup("HAM")


def step_gap(t_grid, gap_t, gap_v):
    idx = np.searchsorted(gap_t, t_grid, side="right") - 1
    idx = np.clip(idx, 0, len(gap_v) - 1)
    return gap_v[idx]


b_ant_gap = step_gap(b_t, ant_gap_t, ant_gap_v)
b_ham_gap = step_gap(b_t, ham_gap_t, ham_gap_v)

# ─────────────────────────────────────────────────────────────────────────
# Segment C — Leclerc's fastest lap (signature pace), compressed ~2.3x
# ─────────────────────────────────────────────────────────────────────────
C_DURATION = h3_lec["TimeSeconds"].max()
C_TARGET_SECONDS = 40
C_FRAMES = C_TARGET_SECONDS * FPS
c_t = np.linspace(0, C_DURATION, C_FRAMES)
c_lec = interp_driver(h3_lec, "TimeSeconds", c_t)

corner_frame = {}
for _, row in corners[corners["Name"].isin(SIGNATURE_CORNERS)].iterrows():
    idx = int(np.argmin(np.abs(c_lec["dist"] - row["Distance"])))
    corner_frame[row["Name"]] = idx

# ─────────────────────────────────────────────────────────────────────────
# Frame index bookkeeping
# ─────────────────────────────────────────────────────────────────────────
A_START, A_END = 0, A_FRAMES
B_START = A_END
C_START = B_START + B_FRAMES
TOTAL_FRAMES = C_START + C_FRAMES

TITLES = {
    "A": ("1 — LIGHTS OUT, TURN ONE", "Leclerc (P2) beats Antonelli (P1) into Abbey"),
    "B": ("2 — THE COLLAPSE", "Lap 41-42: Antonelli's issue costs him track position"),
    "C": ("3 — SIGNATURE LAP", "Leclerc's fastest lap of the race, 1:32.871"),
}

# ─────────────────────────────────────────────────────────────────────────
# Figure setup — IG Reels portrait, 1080x1920 (9:16) at 120dpi
# ─────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(9, 16), dpi=120, facecolor=BG)

# Track lives in a centered box (not the full figure) so title/HUD text has
# dedicated bands above, below, and beside it instead of overlapping it.
# All critical text is kept within y=0.15-0.85 (IG's center-4:5 safe zone,
# 1080x1350 of the 1080x1920 frame) so it survives feed-crop and isn't
# hidden behind IG's caption/username/engagement-button overlays.
TRACK_BOX = [0.286, 0.34, 0.428, 0.41]  # [left, bottom, width, height]
ax = fig.add_axes(TRACK_BOX)
ax.set_facecolor(BG)

ax.plot(ghost_x, ghost_y, color="white", alpha=0.08, linewidth=6, zorder=1)
ax.set_aspect("equal")
ax.autoscale_view()
ax.axis("off")

TRAIL_LEN = 18  # ~0.6s comet tail at 30fps


def make_car(color):
    glow, = ax.plot([], [], "o", color=color, markersize=20, alpha=0.20, zorder=4)
    dot, = ax.plot([], [], "o", color="white", markeredgecolor=color, markeredgewidth=3,
                   markersize=8, zorder=5)
    trail, = ax.plot([], [], "-", color=color, alpha=0.55, linewidth=4, zorder=3)
    return glow, dot, trail


lec_glow, lec_dot, lec_trail = make_car(COL_LEC)
ant_glow, ant_dot, ant_trail = make_car(COL_ANT)
ham_glow, ham_dot, ham_trail = make_car(COL_HAM)

# Pulsing distress ring around Antonelli's car during the collapse phase
ant_pulse, = ax.plot([], [], "o", markerfacecolor="none", markeredgecolor="#FF3333",
                      markeredgewidth=2, markersize=0, zorder=6)

# Corner label markers for Highlight 3
corner_texts = {}
for name in SIGNATURE_CORNERS:
    row = corners[corners["Name"] == name].iloc[0]
    cx, cy = row["X"], row["Y"]
    corner_texts[name] = ax.text(cx, cy, name.upper(), color=DIM, fontsize=11,
                                  fontweight="bold", fontfamily="monospace",
                                  ha="center", va="center", zorder=6,
                                  bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec=DIM, alpha=0.0))

# ── Overlay HUD (figure-fraction coordinates, full canvas) ────────────────
# Everything critical (text, bars, labels) is kept within y=0.15-0.85 —
# IG's center-4:5 safe zone for a 1080x1920 Reel. Bands: title 0.785-0.85,
# track box 0.34-0.75, HUD 0.15-0.32 (bars, legend, gap, event tag).
overlay_ax = fig.add_axes([0, 0, 1, 1])
overlay_ax.set_facecolor("none")
overlay_ax.axis("off")
overlay_ax.set_xlim(0, 1)
overlay_ax.set_ylim(0, 1)
overlay_ax.set_zorder(10)

title_text = overlay_ax.text(0.06, 0.83, "", color="white", fontsize=20, fontweight="bold",
                              va="top", fontfamily="monospace")
subtitle_text = overlay_ax.text(0.06, 0.79, "", color="#aaaaaa", fontsize=10.5,
                                 va="top", fontfamily="monospace", wrap=True)
event_text = overlay_ax.text(0.5, 0.155, "", color="#ffcc00", fontsize=11, fontweight="bold",
                              va="bottom", ha="center", fontfamily="monospace")

# Lap counter / gap readout — right margin, vertically centered beside the track
lap_text = overlay_ax.text(0.885, 0.565, "", color="white", fontsize=13,
                            va="top", ha="center", fontfamily="monospace")
gap_ant_text = overlay_ax.text(0.885, 0.52, "", color=COL_ANT, fontsize=12,
                                va="top", ha="center", fontfamily="monospace")
gap_ham_text = overlay_ax.text(0.885, 0.485, "", color=COL_HAM, fontsize=12,
                                va="top", ha="center", fontfamily="monospace")

# Legend (driver name tags) — left margin, vertically centered beside the track
lec_tag = overlay_ax.text(0.03, 0.565, "● LECLERC", color=COL_LEC, fontsize=11,
                           fontweight="bold", fontfamily="monospace", alpha=0.0)
ant_tag = overlay_ax.text(0.03, 0.53, "● ANTONELLI", color=COL_ANT, fontsize=11,
                           fontweight="bold", fontfamily="monospace", alpha=0.0)
ham_tag = overlay_ax.text(0.03, 0.495, "● HAMILTON", color=COL_HAM, fontsize=11,
                           fontweight="bold", fontfamily="monospace", alpha=0.0)

# ── Speed / brake bar meters, bottom HUD band ──────────────────────────────
BAR_W, BAR_H_MAX, BAR_GAP = 0.045, 0.08, 0.06
BAR_Y0 = 0.185


def make_bar_pair(x0, speed_color="#4CAF50"):
    outline_spd = Rectangle((x0, BAR_Y0), BAR_W, BAR_H_MAX, fill=False,
                             edgecolor="#333333", linewidth=1)
    outline_brk = Rectangle((x0 + BAR_W + BAR_GAP, BAR_Y0), BAR_W, BAR_H_MAX, fill=False,
                             edgecolor="#333333", linewidth=1)
    overlay_ax.add_patch(outline_spd)
    overlay_ax.add_patch(outline_brk)
    spd = Rectangle((x0, BAR_Y0), BAR_W, 0, color=speed_color, zorder=8)
    brk = Rectangle((x0 + BAR_W + BAR_GAP, BAR_Y0), BAR_W, 0, color="#F44336", zorder=8)
    overlay_ax.add_patch(spd)
    overlay_ax.add_patch(brk)
    label_spd = overlay_ax.text(x0 + BAR_W / 2, BAR_Y0 - 0.012, "Speed", color="#aaaaaa",
                                 fontsize=8, ha="center", va="top", fontfamily="monospace")
    label_brk = overlay_ax.text(x0 + BAR_W + BAR_GAP + BAR_W / 2, BAR_Y0 - 0.012, "Brake",
                                 color="#aaaaaa", fontsize=8, ha="center", va="top",
                                 fontfamily="monospace")
    val_text = overlay_ax.text(x0 + BAR_W / 2, BAR_Y0 + BAR_H_MAX + 0.012, "", color="white",
                                fontsize=9, ha="center", va="bottom", fontfamily="monospace")
    return spd, brk, val_text, outline_spd, outline_brk, label_spd, label_brk


BAR_X0_ANT = 0.14
BAR_X0_LEC = 0.58
(lec_speed_bar, lec_brake_bar, lec_speed_val,
 lec_outline_spd, lec_outline_brk, lec_label_spd, lec_label_brk) = make_bar_pair(
    BAR_X0_LEC, speed_color="#4CAF50")
(ant_speed_bar, ant_brake_bar, ant_speed_val,
 ant_outline_spd, ant_outline_brk, ant_label_spd, ant_label_brk) = make_bar_pair(
    BAR_X0_ANT, speed_color="#00D7B6")
ANT_BAR_PARTS = [ant_speed_bar, ant_brake_bar, ant_outline_spd, ant_outline_brk,
                 ant_label_spd, ant_label_brk]

lec_bar_label = overlay_ax.text(BAR_X0_LEC + BAR_W + BAR_GAP / 2, BAR_Y0 + BAR_H_MAX + 0.045,
                                 "LECLERC", color=COL_LEC, fontsize=10, fontweight="bold",
                                 ha="center", va="bottom", fontfamily="monospace")
ant_bar_label = overlay_ax.text(BAR_X0_ANT + BAR_W + BAR_GAP / 2, BAR_Y0 + BAR_H_MAX + 0.045,
                                 "ANTONELLI", color=COL_ANT, fontsize=10, fontweight="bold",
                                 ha="center", va="bottom", fontfamily="monospace", alpha=0.0)


def set_alpha_group(pairs):
    for artist, a in pairs:
        artist.set_alpha(a)


def clear_car(glow, dot, trail):
    glow.set_data([], [])
    dot.set_data([], [])
    trail.set_data([], [])


def update(frame):
    if frame < B_START:
        seg = "A"
    elif frame < C_START:
        seg = "B"
    else:
        seg = "C"

    if seg == "A":
        i = frame - A_START
        title_text.set_text(TITLES["A"][0])
        subtitle_text.set_text(TITLES["A"][1])
        lap_text.set_text("")
        gap_ant_text.set_text("")
        gap_ham_text.set_text("")
        set_alpha_group([(lec_tag, 1.0), (ant_tag, 1.0), (ham_tag, 0.0)])

        lo = max(0, i - TRAIL_LEN)
        lec_dot.set_data([a_lec["x"][i]], [a_lec["y"][i]])
        lec_glow.set_data([a_lec["x"][i]], [a_lec["y"][i]])
        lec_trail.set_data(a_lec["x"][lo:i + 1], a_lec["y"][lo:i + 1])
        ant_dot.set_data([a_ant["x"][i]], [a_ant["y"][i]])
        ant_glow.set_data([a_ant["x"][i]], [a_ant["y"][i]])
        ant_trail.set_data(a_ant["x"][lo:i + 1], a_ant["y"][lo:i + 1])
        clear_car(ham_glow, ham_dot, ham_trail)
        ant_pulse.set_markersize(0)

        lec_speed_val.set_text(f"{a_lec['speed'][i]:.0f} km/h")
        lec_speed_bar.set_height(BAR_H_MAX * min(a_lec["speed"][i] / SPEED_MAX, 1.0))
        lec_brake_bar.set_height(BAR_H_MAX if a_lec["brake"][i] > 0.5 else 0.0)

        ant_speed_val.set_text(f"{a_ant['speed'][i]:.0f} km/h")
        ant_speed_bar.set_height(BAR_H_MAX * min(a_ant["speed"][i] / SPEED_MAX, 1.0))
        ant_brake_bar.set_height(BAR_H_MAX if a_ant["brake"][i] > 0.5 else 0.0)
        ant_bar_label.set_alpha(1.0)
        set_alpha_group([(p, 1.0) for p in ANT_BAR_PARTS])
        ant_speed_val.set_alpha(1.0)

        for t in corner_texts.values():
            t.set_alpha(0.0)

        event_text.set_text("LAP 1 · GRID: LECLERC P2, ANTONELLI P1")

    elif seg == "B":
        i = frame - B_START
        title_text.set_text(TITLES["B"][0])
        subtitle_text.set_text(TITLES["B"][1])
        lap_text.set_text(f"LAP {int(b_lap[i]):>2d} / 48")
        gap_ant_text.set_text(f"ANT  +{b_ant_gap[i]:5.1f}s")
        gap_ham_text.set_text(f"HAM  +{b_ham_gap[i]:5.1f}s")
        set_alpha_group([(lec_tag, 1.0), (ant_tag, 1.0), (ham_tag, 1.0)])
        ant_bar_label.set_alpha(0.0)

        lo = max(0, i - TRAIL_LEN)
        for glow, dot, trail, d in [
            (lec_glow, lec_dot, lec_trail, b_lec),
            (ant_glow, ant_dot, ant_trail, b_ant),
            (ham_glow, ham_dot, ham_trail, b_ham),
        ]:
            dot.set_data([d["x"][i]], [d["y"][i]])
            glow.set_data([d["x"][i]], [d["y"][i]])
            trail.set_data(d["x"][lo:i + 1], d["y"][lo:i + 1])

        lec_speed_val.set_text(f"{b_lec['speed'][i]:.0f} km/h")
        lec_speed_bar.set_height(BAR_H_MAX * min(b_lec["speed"][i] / SPEED_MAX, 1.0))
        lec_brake_bar.set_height(BAR_H_MAX if b_lec["brake"][i] > 0.5 else 0.0)
        set_alpha_group([(p, 0.0) for p in ANT_BAR_PARTS])
        ant_speed_val.set_alpha(0.0)

        for t in corner_texts.values():
            t.set_alpha(0.0)

        ant_pulse.set_data([b_ant["x"][i]], [b_ant["y"][i]])
        pulse_phase = i / max(B_FRAMES, 1)
        ant_pulse.set_markersize(16 + 6 * np.sin(pulse_phase * 60))
        event_text.set_text("LAP 41-42 · ANTONELLI FRONT-WING/WHEEL-SHIELD ISSUE (P2 → P6)")

    else:  # seg == "C"
        i = frame - C_START
        title_text.set_text(TITLES["C"][0])
        subtitle_text.set_text(TITLES["C"][1])
        lap_text.set_text("")
        gap_ant_text.set_text("")
        gap_ham_text.set_text("")
        set_alpha_group([(lec_tag, 1.0), (ant_tag, 0.0), (ham_tag, 0.0)])
        ant_bar_label.set_alpha(0.0)
        ant_pulse.set_markersize(0)

        lo = max(0, i - TRAIL_LEN)
        lec_dot.set_data([c_lec["x"][i]], [c_lec["y"][i]])
        lec_glow.set_data([c_lec["x"][i]], [c_lec["y"][i]])
        lec_trail.set_data(c_lec["x"][lo:i + 1], c_lec["y"][lo:i + 1])
        clear_car(ant_glow, ant_dot, ant_trail)
        clear_car(ham_glow, ham_dot, ham_trail)

        lec_speed_val.set_text(f"{c_lec['speed'][i]:.0f} km/h")
        lec_speed_bar.set_height(BAR_H_MAX * min(c_lec["speed"][i] / SPEED_MAX, 1.0))
        lec_brake_bar.set_height(BAR_H_MAX if c_lec["brake"][i] > 0.5 else 0.0)
        set_alpha_group([(p, 0.0) for p in ANT_BAR_PARTS])
        ant_speed_val.set_alpha(0.0)

        for name, cf in corner_frame.items():
            near = abs(i - cf) < 15
            corner_texts[name].set_color("#FFD54F" if near else DIM)
            corner_texts[name].set_fontsize(14 if near else 11)
            corner_texts[name].set_alpha(1.0 if (near or i > cf) else 0.35)

        event_text.set_text("")

    return (lec_glow, lec_dot, lec_trail, ant_glow, ant_dot, ant_trail,
            ham_glow, ham_dot, ham_trail, ant_pulse, title_text, subtitle_text, event_text,
            lap_text, gap_ant_text, gap_ham_text, lec_tag, ant_tag, ham_tag,
            lec_speed_bar, lec_brake_bar, lec_speed_val, ant_bar_label,
            *ANT_BAR_PARTS, *corner_texts.values())


print(f"Rendering {TOTAL_FRAMES} frames at {FPS} fps "
      f"({A_FRAMES}A + {B_FRAMES}B + {C_FRAMES}C, "
      f"{TOTAL_FRAMES / FPS:.1f}s total)...")
ani = animation.FuncAnimation(fig, update, frames=TOTAL_FRAMES, blit=True)

writer = animation.FFMpegWriter(fps=FPS, bitrate=8000,
                                 extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
ani.save(str(OUT_MP4), writer=writer)
print(f"Saved -> {OUT_MP4}")
