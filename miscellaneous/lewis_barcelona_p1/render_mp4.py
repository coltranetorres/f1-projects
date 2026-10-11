#!/usr/bin/env python3
"""
Renders index.html → lewis_barcelona_p1.mp4 (1080x1920, 30fps, 45s)
Uses Playwright for frame capture + ffmpeg for encoding.
"""
import os, subprocess, shutil, sys
from pathlib import Path
from playwright.sync_api import sync_playwright

# ── CONFIG ───────────────────────────────────────────────────────────────────
FPS        = 30
DURATION   = 45        # output video seconds
TOTAL_MS   = 761870    # race window in ms (laps 33–41)
FLASH_MS   = 2000      # how long the P1 flash stays on screen (ms of race time)
OUT_FILE   = Path(__file__).parent / 'lewis_barcelona_p1.mp4'
FRAMES_DIR = Path(__file__).parent / '_frames'

# ── FFMPEG PATH ──────────────────────────────────────────────────────────────
# Try PATH first, then the local extracted copy
def find_ffmpeg():
    for candidate in ['ffmpeg',
                      str(Path(__file__).parent.parent / 'ffmpeg' / 'bin' / 'ffmpeg.exe')]:
        try:
            subprocess.run([candidate, '-version'], capture_output=True, check=True)
            return candidate
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    sys.exit('ffmpeg not found. Add it to PATH or place ffmpeg/bin/ffmpeg.exe next to this repo.')

# ── FRAME RENDER JS ──────────────────────────────────────────────────────────
RENDER_JS = """(raceMs) => {
    if (raceMs < MS.p1) p1Flashed = false;
    redrawStatic();
    const h = driverPos(HAM, raceMs);
    const r = driverPos(RUS, raceMs);
    const rusInPit = raceMs >= MS.lap37 && raceMs < MS.p1;
    drawDriver(h, '#E8002D', 'HAM', false);
    drawDriver(r, '#27F4D2', 'RUS', rusInPit);
    updateUI(raceMs);
    updateTelDOM(raceMs);
    // Manual flash control (setTimeout doesn't work in headless capture)
    const fl   = document.getElementById('flash');
    const ftxt = document.getElementById('flash-text');
    const inFlash = raceMs >= MS.p1 && raceMs <= MS.p1 + %d;
    fl.classList.toggle('show', inFlash);
    if (ftxt) {
        ftxt.style.opacity   = inFlash ? '1' : '0';
        ftxt.style.transform = inFlash ? 'scale(1)' : 'scale(0.7)';
    }
}""" % FLASH_MS

def main():
    ffmpeg = find_ffmpeg()
    FRAMES_DIR.mkdir(exist_ok=True)
    html_path = Path(__file__).parent / 'index.html'
    total_frames = FPS * DURATION

    print(f'Rendering {total_frames} frames ({DURATION}s @ {FPS}fps) — this takes ~4–6 min ...')

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={'width': 1080, 'height': 1920})
        page.goto(html_path.as_uri())
        page.wait_for_load_state('networkidle')
        page.wait_for_timeout(800)   # let canvas do initial paint

        for i in range(total_frames):
            race_ms = int(i / total_frames * TOTAL_MS)
            page.evaluate(RENDER_JS, race_ms)
            page.screenshot(path=str(FRAMES_DIR / f'f{i:05d}.png'),
                            full_page=False)

            if i % FPS == 0:
                pct = 100 * i // total_frames
                print(f'  [{pct:3d}%] frame {i}/{total_frames}  race_ms={race_ms}')

        browser.close()

    print(f'\nEncoding with ffmpeg → {OUT_FILE.name} ...')
    subprocess.run([
        ffmpeg, '-y',
        '-framerate', str(FPS),
        '-i', str(FRAMES_DIR / 'f%05d.png'),
        '-c:v', 'libx264',
        '-preset', 'slow',
        '-crf', '16',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        '-vf', 'scale=1080:1920',
        str(OUT_FILE),
    ], check=True)

    shutil.rmtree(FRAMES_DIR)
    size_mb = OUT_FILE.stat().st_size / 1024 / 1024
    print(f'\nDone!  {OUT_FILE}  ({size_mb:.1f} MB)')

if __name__ == '__main__':
    main()
