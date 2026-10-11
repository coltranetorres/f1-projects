#!/usr/bin/env python3
import json, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open('data.json') as f:
    raw = json.load(f)

# Telemetry for both drivers over the full window, thinned to every 2nd point
def prep_tel(records):
    return [{'t': r['date'], 'spd': r.get('speed', 0),
             'brk': r.get('brake', 0), 'thr': r.get('throttle', 0),
             'gear': r.get('n_gear', 0)}
            for r in records][::2]

tel_ham_full = prep_tel(raw.get('tel_ham_full', []))
tel_rus_full = prep_tel(raw.get('tel_rus_full', []))

laps_ham = {l['lap_number']: l for l in raw['laps_ham']}
laps_rus = {l['lap_number']: l for l in raw['laps_rus']}

# Focus window: laps 33–41
T_START = '2026-06-14T13:48:12'
T_END   = '2026-06-14T14:00:54'

def in_window(p):
    return T_START <= p['date'][:19] <= T_END

ham_frames = [{'t': p['date'], 'x': p['x'], 'y': p['y']}
              for p in raw['loc_ham'] if in_window(p)][::2]
rus_frames = [{'t': p['date'], 'x': p['x'], 'y': p['y']}
              for p in raw['loc_rus'] if in_window(p)][::2]

# Speed heatmap: map lap-35 telemetry onto circuit outline by proportion
circuit = raw['circuit']
tel     = raw['tel_ham']
n_c, n_t = len(circuit), len(tel)
heatmap = []
for i, pt in enumerate(circuit):
    ti  = min(int(i * n_t / n_c), n_t - 1)
    t   = tel[ti]
    heatmap.append({'x': pt['x'], 'y': pt['y'],
                    'spd': t.get('speed', 200),
                    'brk': 1 if t.get('brake', 0) > 0 else 0,
                    'thr': t.get('throttle', 0)})

# Pace chart data (laps 29–41)
pace = []
for n in range(29, 42):
    h = laps_ham.get(n, {})
    r = laps_rus.get(n, {})
    pace.append({
        'lap':     n,
        'ham':     round(h['lap_duration'], 3) if h.get('lap_duration') else None,
        'rus':     round(r['lap_duration'], 3) if r.get('lap_duration') else None,
        'ham_pit': bool(h.get('is_pit_out_lap', False)),
        'rus_pit': bool(r.get('is_pit_out_lap', False)),
    })

# Milestone real-ms offsets (from T_START = 13:48:12)
# Lap 36 start → 243 376 ms  (pace collapse)
# Lap 37 start → 324 693 ms  (Russell pits)
# P1 moment   → 408 000 ms  (13:55:00)
# Window end  → 761 870 ms
MILESTONES = {'lap36': 243376, 'lap37': 324693, 'p1': 408000, 'end': 761870}
LAP_STARTS = {
    33: 162,   34: 81334, 35: 162321, 36: 243376,
    37: 324693, 38: 405590, 39: 486917, 40: 568862, 41: 650557
}

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=1080">
<title>Hamilton P1 · Barcelona GP 2026</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
html,body{{width:1080px;height:1920px;overflow:hidden;background:#07070f;
  font-family:'Helvetica Neue',Arial,sans-serif;color:#fff}}

#root{{display:flex;flex-direction:column;width:1080px;height:1920px}}

/* ── HEADER ── */
#hdr{{background:linear-gradient(180deg,#180000 0%,#0a0a18 100%);
  border-bottom:3px solid #E8002D;display:flex;flex-direction:column;
  align-items:center;justify-content:center;padding:28px 32px 22px;gap:10px;
  flex:0 0 160px}}
.pills{{display:flex;gap:10px;align-items:center}}
.pill{{background:#E8002D;color:#fff;padding:5px 14px;border-radius:3px;
  font-weight:900;font-size:16px;letter-spacing:2px;text-transform:uppercase}}
.pill.teal{{background:#27F4D2;color:#000}}
#hdr h1{{font-size:32px;font-weight:900;letter-spacing:1px;text-align:center;line-height:1.15}}
#hdr .sub{{font-size:16px;color:#555;letter-spacing:1px;text-transform:uppercase}}

/* ── CIRCUIT ── */
#cv-wrap{{position:relative;flex:0 0 1090px;background:#07070f}}
#cv{{display:block;width:100%;height:100%}}

/* Lap label overlay */
#lap-label{{position:absolute;top:20px;left:28px;
  font-size:18px;font-weight:800;letter-spacing:1px;color:#888;text-transform:uppercase}}
#lap-label span{{font-size:52px;color:#fff;font-weight:900;display:block;line-height:1}}

/* Speed legend */
#spd-legend{{position:absolute;bottom:20px;left:28px;display:flex;flex-direction:column;gap:5px}}
.leg-row{{display:flex;align-items:center;gap:9px;font-size:14px;color:#666}}
.leg-swatch{{width:28px;height:7px;border-radius:3px}}

/* P1 flash */
#flash{{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
  background:rgba(232,0,45,.08);opacity:0;pointer-events:none;transition:opacity .4s}}
#flash.show{{opacity:1}}
#flash-text{{font-size:140px;font-weight:900;letter-spacing:8px;color:#fff;
  text-shadow:0 0 80px #E8002D,0 0 160px #E8002D;
  opacity:0;transform:scale(.7);transition:opacity .35s,transform .35s}}
#flash.show #flash-text{{opacity:1;transform:scale(1)}}

/* ── DRIVER STATS (side by side) ── */
#stats{{display:flex;flex:0 0 220px;border-top:1px solid #141428;border-bottom:1px solid #141428}}
.drv-card{{flex:1;padding:22px 28px;display:flex;flex-direction:column;gap:6px}}
.drv-card:first-child{{border-right:1px solid #141428}}
.drv-top{{display:flex;align-items:center;gap:10px}}
.drv-dot{{width:12px;height:12px;border-radius:50%;flex-shrink:0}}
.drv-abbr{{font-size:16px;font-weight:900;letter-spacing:2px}}
.pos-badge{{margin-left:auto;font-size:20px;font-weight:900;transition:color .5s}}
.drv-time{{font-size:46px;font-weight:900;font-variant-numeric:tabular-nums;line-height:1;margin-top:4px}}
.drv-meta{{font-size:13px;color:#555;letter-spacing:.5px;margin-top:2px}}

/* ── TELEMETRY BARS ── */
#tel-section{{display:flex;flex:0 0 340px;background:#09091a;border-bottom:1px solid #141428}}
.tel-half{{flex:1;display:flex;flex-direction:column;align-items:center;
  justify-content:flex-end;padding:16px 0 20px;gap:10px}}
.tel-half:first-child{{border-right:1px solid #141428}}
.tel-driver-lbl{{font-size:15px;font-weight:900;letter-spacing:2px;margin-bottom:4px}}
.tel-bars-row{{display:flex;gap:14px;align-items:flex-end;height:200px}}
.tel-col{{display:flex;flex-direction:column;align-items:center;gap:5px}}
.tel-val{{font-size:13px;font-weight:700;font-variant-numeric:tabular-nums;color:#ccc;
  min-height:18px;text-align:center}}
.tel-track{{width:44px;flex:1;background:#13132a;border-radius:5px;
  position:relative;overflow:hidden}}
.tel-fill{{position:absolute;bottom:0;left:0;right:0;border-radius:5px;
  transition:height .08s linear}}
.tel-col-lbl{{font-size:11px;color:#444;letter-spacing:1px;text-transform:uppercase}}
.tel-gear{{font-size:28px;font-weight:900;color:#333;margin-top:4px;font-variant-numeric:tabular-nums}}

/* ── TIMELINE ── */
#tl{{flex:0 0 110px;background:#0b0b18;border-top:1px solid #141428;
  display:flex;align-items:center;padding:0 28px;gap:20px}}
#play{{width:44px;height:44px;background:#E8002D;border:none;border-radius:50%;
  cursor:pointer;color:#fff;font-size:18px;display:flex;align-items:center;
  justify-content:center;flex-shrink:0;padding:0}}
#play:hover{{background:#ff1a3a}}
#tl-lap{{font-size:15px;font-weight:800;letter-spacing:.5px;width:110px;flex-shrink:0}}
#tl-track{{flex:1;height:6px;background:#141428;border-radius:3px;cursor:pointer;position:relative}}
#tl-prog{{position:absolute;left:0;top:0;bottom:0;background:#E8002D;border-radius:3px;pointer-events:none}}
#tl-thumb{{position:absolute;top:50%;width:16px;height:16px;background:#fff;
  border:2px solid #E8002D;border-radius:50%;transform:translate(-50%,-50%);pointer-events:none}}
.spd-btn{{background:#141428;border:1px solid #222;color:#666;padding:5px 12px;
  border-radius:3px;cursor:pointer;font-size:13px;font-family:inherit;letter-spacing:.5px}}
.spd-btn.on{{background:#222;color:#ddd;border-color:#444}}
</style>
</head>
<body>
<div id="root">

<!-- HEADER -->
<div id="hdr">
  <div class="pills">
    <div class="pill">Ferrari</div>
    <div class="pill teal">Medium Masterclass</div>
  </div>
  <h1>LEWIS HAMILTON<br>BARCELONA GP 2026</h1>
  <div class="sub">Laps 33 – 41 · The overcut that won the race</div>
</div>

<!-- CIRCUIT -->
<div id="cv-wrap">
  <canvas id="cv"></canvas>
  <div id="lap-label">Lap<span id="lap-num">33</span></div>
  <div id="spd-legend">
    <div class="leg-row"><div class="leg-swatch" style="background:#0060FF"></div>Braking / &lt;120 km/h</div>
    <div class="leg-row"><div class="leg-swatch" style="background:#00CFFF"></div>120 – 220 km/h</div>
    <div class="leg-row"><div class="leg-swatch" style="background:#00FF88"></div>220 – 290 km/h</div>
    <div class="leg-row"><div class="leg-swatch" style="background:#FFD600"></div>290 – 320 km/h</div>
    <div class="leg-row"><div class="leg-swatch" style="background:#FF2D00"></div>320+ km/h</div>
  </div>
  <div id="flash"><div id="flash-text">P1</div></div>
</div>

<!-- DRIVER STATS -->
<div id="stats">
  <div class="drv-card">
    <div class="drv-top">
      <div class="drv-dot" style="background:#E8002D"></div>
      <div class="drv-abbr" style="color:#E8002D">HAM</div>
      <div class="pos-badge" id="ham-pos" style="color:#888">P2</div>
    </div>
    <div class="drv-time" id="ham-t">—</div>
    <div class="drv-meta" id="ham-m">MEDIUM · ON TRACK</div>
  </div>
  <div class="drv-card">
    <div class="drv-top">
      <div class="drv-dot" style="background:#27F4D2"></div>
      <div class="drv-abbr" style="color:#27F4D2">RUS</div>
      <div class="pos-badge" id="rus-pos" style="color:#FFD700">P1</div>
    </div>
    <div class="drv-time" id="rus-t">—</div>
    <div class="drv-meta" id="rus-m">HARD · ON TRACK</div>
  </div>
</div>

<!-- TELEMETRY BARS -->
<div id="tel-section">
  <div class="tel-half">
    <div class="tel-driver-lbl" style="color:#E8002D">HAM</div>
    <div class="tel-bars-row">
      <div class="tel-col">
        <div class="tel-val" id="hspd-val">0</div>
        <div class="tel-track"><div class="tel-fill" id="hspd-bar" style="height:0%;background:#E8002D"></div></div>
        <div class="tel-col-lbl">SPD</div>
      </div>
      <div class="tel-col">
        <div class="tel-val" id="hbrk-val"></div>
        <div class="tel-track"><div class="tel-fill" id="hbrk-bar" style="height:0%;background:#FF4400"></div></div>
        <div class="tel-col-lbl">BRK</div>
      </div>
    </div>
    <div class="tel-gear" id="hgear">G—</div>
  </div>
  <div class="tel-half">
    <div class="tel-driver-lbl" style="color:#27F4D2">RUS</div>
    <div class="tel-bars-row">
      <div class="tel-col">
        <div class="tel-val" id="rspd-val">0</div>
        <div class="tel-track"><div class="tel-fill" id="rspd-bar" style="height:0%;background:#27F4D2"></div></div>
        <div class="tel-col-lbl">SPD</div>
      </div>
      <div class="tel-col">
        <div class="tel-val" id="rbrk-val"></div>
        <div class="tel-track"><div class="tel-fill" id="rbrk-bar" style="height:0%;background:#FF4400"></div></div>
        <div class="tel-col-lbl">BRK</div>
      </div>
    </div>
    <div class="tel-gear" id="rgear">G—</div>
  </div>
</div>

<!-- TIMELINE -->
<div id="tl">
  <button id="play" onclick="togglePlay()">&#9654;</button>
  <div id="tl-lap">LAP 33 / 66</div>
  <div id="tl-track" onclick="scrubClick(event)">
    <div id="tl-prog" style="width:0%"></div>
    <div id="tl-thumb" style="left:0%"></div>
  </div>
  <div style="display:flex;gap:6px">
    <button class="spd-btn" onclick="setSpd(5)">5×</button>
    <button class="spd-btn on" id="s10" onclick="setSpd(10)">10×</button>
    <button class="spd-btn" id="s20" onclick="setSpd(20)">20×</button>
    <button class="spd-btn" id="s40" onclick="setSpd(40)">40×</button>
  </div>
</div>

</div><!-- #root -->

<script>
// ─── DATA ────────────────────────────────────────────────────────────────────
const HAM = {json.dumps(ham_frames)};
const RUS = {json.dumps(rus_frames)};
const HM  = {json.dumps(heatmap)};   // circuit + speed heatmap
const PACE = {json.dumps(pace)};
const MS = {json.dumps(MILESTONES)};
const LAP_STARTS = {json.dumps(LAP_STARTS)};
const TEL_HAM = {json.dumps(tel_ham_full)};
const TEL_RUS = {json.dumps(tel_rus_full)};

// ─── CANVAS SETUP ────────────────────────────────────────────────────────────
const cv   = document.getElementById('cv');
const ctx  = cv.getContext('2d');
const wrap = document.getElementById('cv-wrap');

function resizeCanvas() {{
  cv.width  = wrap.clientWidth;
  cv.height = wrap.clientHeight;
  redrawStatic();
}}

// coordinate transform
let SC, OX, OY;
function computeTransform() {{
  const xs = HM.map(p=>p.x), ys = HM.map(p=>p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const W = cv.width  - 80;
  const H = cv.height - 60;
  SC = Math.min(W / (maxX - minX), H / (maxY - minY));
  OX = (cv.width  - (maxX - minX) * SC) / 2 - minX * SC;
  OY = (cv.height - (maxY - minY) * SC) / 2 + maxY * SC;
}}

function tx(x) {{ return x * SC + OX; }}
function ty(y) {{ return -y * SC + OY; }}

// ─── SPEED → COLOR ───────────────────────────────────────────────────────────
function spdColor(s, brk) {{
  if (brk) return '#0060FF';
  const t = Math.max(0, Math.min(1, (s - 80) / 260));
  if (t < .25) {{
    const u = t/.25;
    return `rgb(${{lerp(0,0,u)}},${{lerp(96,207,u)}},${{lerp(255,255,u)}})`;
  }} else if (t < .55) {{
    const u = (t-.25)/.30;
    return `rgb(${{lerp(0,0,u)}},${{lerp(207,255,u)}},${{lerp(255,136,u)}})`;
  }} else if (t < .80) {{
    const u = (t-.55)/.25;
    return `rgb(${{lerp(0,255,u)}},${{lerp(255,214,u)}},${{lerp(136,0,u)}})`;
  }} else {{
    const u = (t-.80)/.20;
    return `rgb(255,${{lerp(214,45,u)}},0)`;
  }}
}}

function lerp(a,b,t) {{ return Math.round(a + (b-a)*t); }}

// ─── STATIC CIRCUIT DRAW ─────────────────────────────────────────────────────
function redrawStatic() {{
  computeTransform();
  ctx.clearRect(0, 0, cv.width, cv.height);

  // Outer glow / track base (thick)
  ctx.save();
  ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  ctx.lineWidth = 18;
  ctx.strokeStyle = '#1a1a2a';
  ctx.beginPath();
  HM.forEach((p,i) => i===0 ? ctx.moveTo(tx(p.x),ty(p.y)) : ctx.lineTo(tx(p.x),ty(p.y)));
  ctx.closePath();
  ctx.stroke();

  // Track surface
  ctx.lineWidth = 13;
  ctx.strokeStyle = '#252535';
  ctx.beginPath();
  HM.forEach((p,i) => i===0 ? ctx.moveTo(tx(p.x),ty(p.y)) : ctx.lineTo(tx(p.x),ty(p.y)));
  ctx.closePath();
  ctx.stroke();
  ctx.restore();

  // Speed heatmap (thin coloured line)
  ctx.save();
  ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  for (let i=0; i<HM.length-1; i++) {{
    const a=HM[i], b=HM[i+1];
    ctx.beginPath();
    ctx.lineWidth = 5;
    ctx.strokeStyle = spdColor(a.spd, a.brk);
    ctx.moveTo(tx(a.x),ty(a.y));
    ctx.lineTo(tx(b.x),ty(b.y));
    ctx.stroke();
  }}
  ctx.restore();

  // Start/finish line
  const sf = HM[0];
  ctx.save();
  ctx.fillStyle = '#fff';
  ctx.beginPath();
  ctx.arc(tx(sf.x), ty(sf.y), 6, 0, Math.PI*2);
  ctx.fill();
  ctx.fillStyle = '#aaa';
  ctx.font = '10px Helvetica Neue';
  ctx.fillText('S/F', tx(sf.x)+8, ty(sf.y)+4);
  ctx.restore();
}}

// ─── ANIMATION STATE ─────────────────────────────────────────────────────────
let playing = false;
let playSpd = 10;
let animStart = null;   // performance.now() when play last started
let elapsedPause = 0;   // accumulated real-ms at last pause
let rafId = null;
let p1Flashed = false;

// Pre-parse timestamps
function msOf(iso) {{
  return new Date(iso).getTime();
}}
HAM.forEach(f => f.ms = msOf(f.t));
RUS.forEach(f => f.ms = msOf(f.t));

const T0 = HAM[0].ms;
const TOTAL = MS.end;  // 761870 ms of race time

// ─── INTERPOLATE TELEMETRY ───────────────────────────────────────────────────
function driverTel(telFrames, raceMs) {{
  const target = T0 + raceMs;
  let lo=0, hi=telFrames.length-1;
  while (lo < hi-1) {{
    const mid=(lo+hi)>>1;
    new Date(telFrames[mid].t).getTime() <= target ? (lo=mid) : (hi=mid);
  }}
  return telFrames[lo];
}}

// Pre-parse tel timestamps once
TEL_HAM.forEach(f => f.ms = new Date(f.t).getTime());
TEL_RUS.forEach(f => f.ms = new Date(f.t).getTime());

function driverTelFast(telFrames, raceMs) {{
  const target = T0 + raceMs;
  let lo=0, hi=telFrames.length-1;
  while (lo < hi-1) {{
    const mid=(lo+hi)>>1;
    telFrames[mid].ms <= target ? (lo=mid) : (hi=mid);
  }}
  return telFrames[lo];
}}

// ─── INTERPOLATE POSITION ────────────────────────────────────────────────────
function driverPos(frames, raceMs) {{
  const target = T0 + raceMs;
  let lo=0, hi=frames.length-1;
  while (lo < hi-1) {{
    const mid = (lo+hi)>>1;
    frames[mid].ms <= target ? (lo=mid) : (hi=mid);
  }}
  if (lo >= frames.length-1) return frames[frames.length-1];
  const f0=frames[lo], f1=frames[lo+1];
  const t = Math.max(0, Math.min(1, (target-f0.ms)/(f1.ms-f0.ms)));
  return {{ x: f0.x+(f1.x-f0.x)*t, y: f0.y+(f1.y-f0.y)*t }};
}}

// ─── DRIVER DOT ──────────────────────────────────────────────────────────────
function drawDriver(pos, color, label, inPit) {{
  if (!pos) return;
  const sx = tx(pos.x), sy = ty(pos.y);
  if (inPit) {{
    // dashed ghost when in pit
    ctx.save();
    ctx.globalAlpha = 0.3;
    ctx.beginPath();
    ctx.arc(sx, sy, 10, 0, Math.PI*2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.restore();
    return;
  }}
  // Glow
  ctx.save();
  ctx.shadowColor = color;
  ctx.shadowBlur  = 18;
  ctx.beginPath();
  ctx.arc(sx, sy, 9, 0, Math.PI*2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.restore();
  // White border
  ctx.save();
  ctx.beginPath();
  ctx.arc(sx, sy, 9, 0, Math.PI*2);
  ctx.strokeStyle = '#fff';
  ctx.lineWidth   = 2;
  ctx.stroke();
  ctx.restore();
  // Label
  ctx.save();
  ctx.fillStyle = '#fff';
  ctx.font = 'bold 9px Helvetica Neue';
  ctx.textAlign = 'center';
  ctx.fillText(label, sx, sy + 20);
  ctx.restore();
}}

// ─── TELEMETRY BARS (DOM) ────────────────────────────────────────────────────
function updateTelDOM(raceMs) {{
  const ht = driverTelFast(TEL_HAM, raceMs);
  const rt = driverTelFast(TEL_RUS, raceMs);
  if (ht) {{
    document.getElementById('hspd-bar').style.height = Math.max(1,(ht.spd/350*100))+'%';
    document.getElementById('hbrk-bar').style.height = Math.max(1,(ht.brk/100*100))+'%';
    document.getElementById('hbrk-bar').style.background = ht.brk > 10 ? '#FF4400' : '#2a1010';
    document.getElementById('hspd-val').textContent = ht.spd+' km/h';
    document.getElementById('hbrk-val').textContent = ht.brk > 0 ? ht.brk+'%' : '';
    document.getElementById('hgear').textContent = 'G'+ht.gear;
  }}
  if (rt) {{
    document.getElementById('rspd-bar').style.height = Math.max(1,(rt.spd/350*100))+'%';
    document.getElementById('rbrk-bar').style.height = Math.max(1,(rt.brk/100*100))+'%';
    document.getElementById('rbrk-bar').style.background = rt.brk > 10 ? '#FF4400' : '#2a1010';
    document.getElementById('rspd-val').textContent = rt.spd+' km/h';
    document.getElementById('rbrk-val').textContent = rt.brk > 0 ? rt.brk+'%' : '';
    document.getElementById('rgear').textContent = 'G'+rt.gear;
  }}
}}

// ─── CURRENT LAP ─────────────────────────────────────────────────────────────
function currentLap(raceMs) {{
  let lap = 33;
  for (const [l, startMs] of Object.entries(LAP_STARTS)) {{
    if (raceMs >= startMs) lap = parseInt(l);
  }}
  return lap;
}}

// ─── UI UPDATE ───────────────────────────────────────────────────────────────
const paceByLap = {{}};
PACE.forEach(p => paceByLap[p.lap] = p);

function fmt(s) {{
  if (!s || s > 200) return '—';
  const m = Math.floor(s/60);
  const ss = (s % 60).toFixed(3).padStart(6,'0');
  return m>0 ? `${{m}}:${{ss}}` : ss;
}}

function updateUI(raceMs) {{
  const pct = Math.min(raceMs / TOTAL, 1);
  document.getElementById('tl-prog').style.width  = (pct*100)+'%';
  document.getElementById('tl-thumb').style.left  = (pct*100)+'%';

  const lap = currentLap(raceMs);
  document.getElementById('lap-num').textContent = lap;
  document.getElementById('tl-lap').textContent = `LAP ${{lap}} / 66`;

  // Stats
  const pd = paceByLap[lap];
  if (pd) {{
    const hamPit = pd.ham_pit;
    const rusPit = pd.rus_pit;

    document.getElementById('ham-t').textContent = fmt(pd.ham);
    document.getElementById('ham-m').textContent =
      `MEDIUM · ${{hamPit ? 'PIT OUT LAP' : 'ON TRACK'}}`;

    const rusInPit = raceMs >= MS.lap37 && raceMs < MS.lap37 + 30000;
    document.getElementById('rus-t').textContent =
      rusInPit ? 'IN PITS' : fmt(pd.rus);
    document.getElementById('rus-m').textContent =
      `HARD · ${{rusPit ? 'PIT OUT LAP' : rusInPit ? 'SERVICE' : 'ON TRACK'}}`;

  }}

  // Positions
  if (raceMs < MS.lap37) {{
    document.getElementById('ham-pos').textContent = 'P2';
    document.getElementById('ham-pos').style.color = '#888';
    document.getElementById('rus-pos').textContent = 'P1';
    document.getElementById('rus-pos').style.color = '#FFD700';
  }} else if (raceMs < MS.p1) {{
    document.getElementById('ham-pos').textContent = 'P2';
    document.getElementById('ham-pos').style.color = '#888';
    document.getElementById('rus-pos').textContent = 'PIT';
    document.getElementById('rus-pos').style.color = '#27F4D2';
  }} else {{
    document.getElementById('ham-pos').textContent = 'P1';
    document.getElementById('ham-pos').style.color = '#FFD700';
    document.getElementById('rus-pos').textContent = 'P2';
    document.getElementById('rus-pos').style.color = '#888';
  }}

  // P1 flash (once)
  if (raceMs >= MS.p1 && !p1Flashed) {{
    p1Flashed = true;
    const fl = document.getElementById('flash');
    fl.classList.add('show');
    setTimeout(() => fl.classList.remove('show'), 2400);
  }}

}}

// ─── ANIMATION LOOP ──────────────────────────────────────────────────────────
function frame(now) {{
  const raceMs = elapsedPause + (now - animStart) * playSpd;

  if (raceMs >= TOTAL) {{
    // End
    playing = false;
    document.getElementById('play').innerHTML = '&#8635;';
    redrawStatic();
    const h = driverPos(HAM, TOTAL);
    const r = driverPos(RUS, TOTAL);
    drawDriver(h, '#E8002D', 'HAM', false);
    drawDriver(r, '#27F4D2', 'RUS', false);
    updateTelDOM(TOTAL);
    updateUI(TOTAL);
    return;
  }}

  redrawStatic();

  const h = driverPos(HAM, raceMs);
  const r = driverPos(RUS, raceMs);
  const rusInPit = raceMs >= MS.lap37 && raceMs < MS.p1;
  drawDriver(h, '#E8002D', 'HAM', false);
  drawDriver(r, '#27F4D2', 'RUS', rusInPit);
  updateTelDOM(raceMs);

  updateUI(raceMs);
  rafId = requestAnimationFrame(frame);
}}

// ─── CONTROLS ────────────────────────────────────────────────────────────────
function togglePlay() {{
  if (playing) {{
    cancelAnimationFrame(rafId);
    elapsedPause += (performance.now() - animStart) * playSpd;
    playing = false;
    document.getElementById('play').innerHTML = '&#9654;';
  }} else {{
    // Reset if at end
    if (elapsedPause >= TOTAL) {{
      elapsedPause = 0;
      p1Flashed = false;
      document.getElementById('flash').classList.remove('show');
    }}
    animStart = performance.now();
    playing = true;
    document.getElementById('play').innerHTML = '&#9646;&#9646;';
    rafId = requestAnimationFrame(frame);
  }}
}}

function setSpd(s) {{
  if (playing) {{
    // Capture current race time before changing speed
    elapsedPause += (performance.now() - animStart) * playSpd;
    animStart = performance.now();
  }}
  playSpd = s;
  document.querySelectorAll('.spd-btn').forEach(b=>b.classList.remove('on'));
  document.getElementById(`s${{s}}`).classList.add('on');
}}

// add id to spd buttons after setting up
document.querySelectorAll('.spd-btn').forEach(b=>{{
  const txt=b.textContent.replace('×','');
  b.id='s'+txt;
}});

function scrubClick(e) {{
  const rect = document.getElementById('tl-track').getBoundingClientRect();
  const pct  = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  elapsedPause = pct * TOTAL;
  p1Flashed = elapsedPause >= MS.p1;
  if (!p1Flashed) document.getElementById('flash').classList.remove('show');
  animStart = performance.now();
  // single frame render
  const was = playing;
  if (!was) {{
    redrawStatic();
    const h = driverPos(HAM, elapsedPause);
    const r = driverPos(RUS, elapsedPause);
    drawDriver(h, '#E8002D', 'HAM', false);
    const rusInPit = elapsedPause >= MS.lap37 && elapsedPause < MS.p1;
    drawDriver(r, '#27F4D2', 'RUS', rusInPit);
    updateTelDOM(elapsedPause);
    updateUI(elapsedPause);
  }}
}}

// ─── INIT ─────────────────────────────────────────────────────────────────────
window.addEventListener('resize', () => {{ resizeCanvas(); }});
resizeCanvas();

// Initial static frame
redrawStatic();
const h0 = driverPos(HAM, 0);
const r0 = driverPos(RUS, 0);
drawDriver(h0, '#E8002D', 'HAM', false);
drawDriver(r0, '#27F4D2', 'RUS', false);
updateTelDOM(0);
updateUI(0);
</script>
</body>
</html>"""

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'index.html')
with open(out, 'w', encoding='utf-8') as f:
    f.write(HTML)
print(f'Written: {out}')
print(f'HAM frames: {len(ham_frames)}, RUS frames: {len(rus_frames)}, HM: {len(heatmap)}, Pace: {len(pace)}')
