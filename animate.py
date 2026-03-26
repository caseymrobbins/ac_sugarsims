"""
animate.py
----------
Step-by-step simulation animation system.

Two output modes:
  1. generate_animation_html() - Interactive HTML5 player (recommended)
     Self-contained HTML file with canvas rendering and playback controls.
     No dependencies, opens in any browser.

  2. generate_animation_gif() - Matplotlib GIF (slower, larger files)
     Traditional animated GIF using matplotlib FuncAnimation.

The HTML player renders:
  - Workers as colored dots (color = wealth, border = employed/not)
  - Firms as squares (size = n_workers, red border if cartel)
  - Pollution overlay (semi-transparent red)
  - Food density overlay (semi-transparent green, optional)
  - Live metric dashboard (Gini, unemployment, agency floor, HI, etc.)
  - Playback controls: play/pause, speed, step forward/back, scrubber

Usage:
    from animate import generate_animation_html

    # After running simulation:
    generate_animation_html(
        model.animation_frames,
        output_path="results/simulation.html",
        grid_size=80,
    )
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional

import numpy as np


def generate_animation_html(
    frames: List[Dict[str, Any]],
    output_path: str = "results/simulation.html",
    grid_size: int = 80,
    title: str = "Economic Simulation",
    subsample: int = 1,
) -> str:
    """
    Generate a self-contained HTML5 animation player.

    Parameters
    ----------
    frames : list of dicts from collect_animation_frame()
    output_path : where to save the HTML file
    grid_size : simulation grid dimensions (assumed square)
    title : page title
    subsample : use every Nth frame (1 = all frames)

    Returns
    -------
    output_path : the path written to
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # Subsample frames if requested
    if subsample > 1:
        frames = frames[::subsample]

    # Serialize frames to compact JSON
    # Convert numpy types to native Python for JSON serialization
    frames_json = json.dumps(frames, default=_json_serializer)

    html = _ANIMATION_HTML_TEMPLATE.replace("__FRAMES_DATA__", frames_json)
    html = html.replace("__GRID_SIZE__", str(grid_size))
    html = html.replace("__TITLE__", title)
    html = html.replace("__N_FRAMES__", str(len(frames)))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    return output_path


def _json_serializer(obj):
    """Handle numpy types in JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# Matplotlib GIF fallback
# ---------------------------------------------------------------------------

def generate_animation_gif(
    frames: List[Dict[str, Any]],
    output_path: str = "results/simulation.gif",
    grid_size: int = 80,
    fps: int = 10,
    subsample: int = 5,
) -> Optional[str]:
    """
    Generate an animated GIF using matplotlib.
    Slower and larger than HTML but works everywhere.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib.colors import Normalize
    except ImportError:
        print("matplotlib not available, skipping GIF generation")
        return None

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    if subsample > 1:
        frames = frames[::subsample]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [2, 1]})
    ax_grid = axes[0]
    ax_metrics = axes[1]

    norm = Normalize(vmin=0, vmax=500)

    def draw_frame(idx):
        frame = frames[idx]
        ax_grid.clear()
        ax_metrics.clear()

        # Draw pollution background
        if "pollution_grid" in frame:
            poll = np.array(frame["pollution_grid"])
            ax_grid.imshow(
                poll.T, origin="lower", cmap="Reds", alpha=0.3,
                extent=[0, grid_size, 0, grid_size], aspect="auto",
                vmin=0, vmax=max(float(np.max(poll)), 1.0),
            )

        # Draw workers
        workers = frame.get("workers", [])
        if workers:
            xs = [w["x"] for w in workers]
            ys = [w["y"] for w in workers]
            ws = [min(w["wealth"], 500) for w in workers]
            emp = [w["employed"] for w in workers]
            colors = ["#2ecc71" if e else "#e74c3c" for e in emp]
            sizes = [max(3, min(20, w / 25)) for w in ws]
            ax_grid.scatter(xs, ys, c=ws, cmap="YlGnBu", s=sizes,
                           alpha=0.7, edgecolors="none", norm=norm)

        # Draw firms
        firms_data = frame.get("firms", [])
        if firms_data:
            for f in firms_data:
                size = max(30, f["n_workers"] * 15)
                color = "#e74c3c" if f["in_cartel"] else "#3498db"
                ax_grid.scatter(f["x"], f["y"], s=size, c=color,
                               marker="s", alpha=0.8, edgecolors="black", linewidths=1)

        ax_grid.set_xlim(0, grid_size)
        ax_grid.set_ylim(0, grid_size)
        ax_grid.set_title(f"Step {frame.get('step', idx)}", fontsize=14)
        ax_grid.set_aspect("equal")

        # Metrics panel
        overlay = frame.get("overlay", {})
        metrics_text = (
            f"Workers: {overlay.get('n_workers', '?')}\n"
            f"Firms: {overlay.get('n_firms', '?')}\n"
            f"Cartels: {overlay.get('n_cartels', '?')}\n"
            f"\n"
            f"Gini: {overlay.get('gini', 0):.3f}\n"
            f"Unemployment: {overlay.get('unemployment', 0):.1%}\n"
            f"Agency Floor: {overlay.get('agency_floor', 0):.2f}\n"
            f"Horizon Index: {overlay.get('horizon_index', 0.5):.3f}\n"
            f"Firm Floor: {overlay.get('mean_firm_floor', 0):.3f}\n"
        )
        ax_metrics.text(0.1, 0.9, metrics_text, transform=ax_metrics.transAxes,
                       fontsize=12, verticalalignment="top", fontfamily="monospace",
                       bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        ax_metrics.axis("off")

        return []

    anim = animation.FuncAnimation(
        fig, draw_frame, frames=len(frames),
        interval=1000 // fps, blit=False,
    )

    try:
        anim.save(output_path, writer="pillow", fps=fps)
        print(f"Animation saved to {output_path}")
    except Exception as e:
        print(f"Failed to save GIF: {e}")
        return None
    finally:
        plt.close(fig)

    return output_path


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_ANIMATION_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>__TITLE__</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    background: #1a1a2e; color: #eee;
    font-family: 'Segoe UI', system-ui, sans-serif;
    display: flex; flex-direction: column; align-items: center;
    min-height: 100vh; padding: 16px;
}
h1 { font-size: 1.4em; margin-bottom: 12px; color: #e0e0e0; }
.container {
    display: flex; gap: 20px; align-items: flex-start;
    flex-wrap: wrap; justify-content: center;
}
canvas {
    border: 1px solid #333; border-radius: 4px;
    background: #0f0f23;
}
.sidebar {
    width: 280px; background: #16213e; border-radius: 8px;
    padding: 16px; font-size: 0.85em;
}
.sidebar h3 { color: #e94560; margin-bottom: 8px; font-size: 1.1em; }
.metric-row {
    display: flex; justify-content: space-between;
    padding: 4px 0; border-bottom: 1px solid #1a1a3e;
}
.metric-label { color: #aaa; }
.metric-value { color: #eee; font-weight: 600; font-family: monospace; }
.metric-value.good { color: #2ecc71; }
.metric-value.warn { color: #f39c12; }
.metric-value.bad { color: #e74c3c; }
.controls {
    margin-top: 16px; display: flex; flex-direction: column;
    gap: 8px; width: 100%; max-width: 700px;
}
.controls-row {
    display: flex; align-items: center; gap: 8px; justify-content: center;
}
button {
    background: #16213e; color: #eee; border: 1px solid #333;
    border-radius: 4px; padding: 6px 14px; cursor: pointer;
    font-size: 0.9em;
}
button:hover { background: #1a1a3e; border-color: #e94560; }
button.active { background: #e94560; border-color: #e94560; }
input[type="range"] { flex: 1; accent-color: #e94560; }
.step-label { font-family: monospace; min-width: 120px; text-align: center; }
.legend {
    margin-top: 12px; padding-top: 8px; border-top: 1px solid #333;
}
.legend-item {
    display: flex; align-items: center; gap: 6px; margin: 3px 0;
    font-size: 0.8em; color: #aaa;
}
.legend-dot {
    width: 10px; height: 10px; border-radius: 50%; display: inline-block;
}
.legend-sq {
    width: 10px; height: 10px; display: inline-block;
}
.speed-label { font-size: 0.8em; color: #aaa; min-width: 50px; }
</style>
</head>
<body>
<h1>__TITLE__</h1>
<div class="container">
    <canvas id="grid" width="560" height="560"></canvas>
    <div class="sidebar">
        <h3>Dashboard</h3>
        <div id="metrics"></div>
        <div class="legend">
            <div class="legend-item"><span class="legend-dot" style="background:#2ecc71"></span> Employed worker</div>
            <div class="legend-item"><span class="legend-dot" style="background:#e74c3c"></span> Unemployed worker</div>
            <div class="legend-item"><span class="legend-dot" style="background:#f1c40f"></span> In debt</div>
            <div class="legend-item"><span class="legend-sq" style="background:#3498db"></span> Firm</div>
            <div class="legend-item"><span class="legend-sq" style="background:#e94560"></span> Cartel firm</div>
            <div class="legend-item"><span style="color:#ff634780">&#9608;</span> Pollution</div>
        </div>
    </div>
</div>
<div class="controls">
    <div class="controls-row">
        <button id="btnPrev" title="Previous step">&laquo;</button>
        <button id="btnPlay" class="active">&#9654; Play</button>
        <button id="btnNext" title="Next step">&raquo;</button>
        <span class="speed-label" id="speedLabel">1x</span>
        <input type="range" id="speedSlider" min="1" max="60" value="10" style="max-width:120px">
    </div>
    <div class="controls-row">
        <span class="step-label" id="stepLabel">Step 0 / __N_FRAMES__</span>
        <input type="range" id="scrubber" min="0" max="__N_FRAMES__" value="0">
    </div>
</div>

<script>
const FRAMES = __FRAMES_DATA__;
const GRID = __GRID_SIZE__;
const canvas = document.getElementById('grid');
const ctx = canvas.getContext('2d');
const W = canvas.width, H = canvas.height;
const CELL = W / GRID;

let frameIdx = 0;
let playing = true;
let speed = 10; // frames per second
let animId = null;
let lastTime = 0;

// Color helpers
function wealthColor(w) {
    const t = Math.min(w / 400, 1);
    const r = Math.round(30 + 50 * (1 - t));
    const g = Math.round(100 + 155 * t);
    const b = Math.round(80 + 120 * t);
    return `rgb(${r},${g},${b})`;
}

function drawFrame(idx) {
    if (idx < 0 || idx >= FRAMES.length) return;
    const frame = FRAMES[idx];
    ctx.clearRect(0, 0, W, H);

    // Pollution overlay
    if (frame.pollution_grid) {
        const pg = frame.pollution_grid;
        const rows = pg.length, cols = pg[0].length;
        const cw = W / cols, ch = H / rows;
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const v = pg[r][c];
                if (v > 0.1) {
                    const a = Math.min(v / 20, 0.5);
                    ctx.fillStyle = `rgba(255,50,50,${a})`;
                    ctx.fillRect(r * cw, (cols - 1 - c) * ch, cw, ch);
                }
            }
        }
    }

    // Food overlay (very subtle green)
    if (frame.food_grid) {
        const fg = frame.food_grid;
        const rows = fg.length, cols = fg[0].length;
        const cw = W / cols, ch = H / rows;
        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const v = fg[r][c];
                if (v > 5) {
                    const a = Math.min(v / 100, 0.15);
                    ctx.fillStyle = `rgba(46,204,113,${a})`;
                    ctx.fillRect(r * cw, (cols - 1 - c) * ch, cw, ch);
                }
            }
        }
    }

    // Workers
    const workers = frame.workers || [];
    for (const w of workers) {
        const x = w.x * CELL + CELL / 2;
        const y = (GRID - 1 - w.y) * CELL + CELL / 2;
        const radius = Math.max(1.5, Math.min(4, w.wealth / 80));

        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);

        if (w.in_debt) {
            ctx.fillStyle = '#f1c40f';
        } else if (w.employed) {
            ctx.fillStyle = wealthColor(w.wealth);
        } else {
            ctx.fillStyle = '#e74c3c';
        }
        ctx.fill();
    }

    // Firms
    const firms = frame.firms || [];
    for (const f of firms) {
        const x = f.x * CELL;
        const y = (GRID - 1 - f.y) * CELL;
        const size = Math.max(4, Math.min(14, 4 + f.n_workers * 1.5));

        ctx.fillStyle = f.in_cartel ? '#e94560' : '#3498db';
        ctx.fillRect(x, y, size, size);
        ctx.strokeStyle = '#000';
        ctx.lineWidth = 0.5;
        ctx.strokeRect(x, y, size, size);
    }

    // Update dashboard
    const overlay = frame.overlay || {};
    const metricsDiv = document.getElementById('metrics');

    function cls(val, good, bad) {
        if (val <= good) return 'good';
        if (val >= bad) return 'bad';
        return 'warn';
    }
    function clsHi(val, good, bad) {
        if (val >= good) return 'good';
        if (val <= bad) return 'bad';
        return 'warn';
    }

    metricsDiv.innerHTML = [
        row('Workers', overlay.n_workers || '?'),
        row('Firms', overlay.n_firms || '?'),
        row('Cartels', overlay.n_cartels || 0, overlay.n_cartels > 0 ? 'bad' : 'good'),
        '<div style="height:8px"></div>',
        row('Gini', fmt(overlay.gini, 3), cls(overlay.gini, 0.35, 0.6)),
        row('Unemployment', pct(overlay.unemployment), cls(overlay.unemployment, 0.10, 0.25)),
        row('Agency Floor', fmt(overlay.agency_floor, 2), clsHi(overlay.agency_floor, 2, 0.5)),
        row('Horizon Index', fmt(overlay.horizon_index, 3), clsHi(overlay.horizon_index, 0.6, 0.3)),
        row('Firm Floor', fmt(overlay.mean_firm_floor, 3), clsHi(overlay.mean_firm_floor, 0.3, 0.1)),
        '<div style="height:8px"></div>',
        row('Planner Trust', fmt(overlay.trust_planner, 3), clsHi(overlay.trust_planner, 0.5, 0.25)),
        row('System Trust', fmt(overlay.trust_institutional, 3), clsHi(overlay.trust_institutional, 0.4, 0.2)),
    ].join('');

    // Step label
    document.getElementById('stepLabel').textContent =
        `Step ${frame.step || idx} / ${FRAMES.length}`;
    document.getElementById('scrubber').value = idx;
}

function row(label, value, cls) {
    const c = cls ? ` ${cls}` : '';
    return `<div class="metric-row"><span class="metric-label">${label}</span><span class="metric-value${c}">${value}</span></div>`;
}
function fmt(v, d) { return v != null ? Number(v).toFixed(d) : '---'; }
function pct(v) { return v != null ? (Number(v) * 100).toFixed(1) + '%' : '---'; }

// Playback
function tick(timestamp) {
    if (!playing) return;
    const interval = 1000 / speed;
    if (timestamp - lastTime >= interval) {
        lastTime = timestamp;
        if (frameIdx < FRAMES.length - 1) {
            frameIdx++;
            drawFrame(frameIdx);
        } else {
            playing = false;
            document.getElementById('btnPlay').classList.remove('active');
            document.getElementById('btnPlay').innerHTML = '&#9654; Play';
        }
    }
    animId = requestAnimationFrame(tick);
}

document.getElementById('btnPlay').onclick = () => {
    playing = !playing;
    const btn = document.getElementById('btnPlay');
    if (playing) {
        if (frameIdx >= FRAMES.length - 1) frameIdx = 0;
        btn.classList.add('active');
        btn.innerHTML = '&#10074;&#10074; Pause';
        lastTime = 0;
        animId = requestAnimationFrame(tick);
    } else {
        btn.classList.remove('active');
        btn.innerHTML = '&#9654; Play';
        if (animId) cancelAnimationFrame(animId);
    }
};

document.getElementById('btnPrev').onclick = () => {
    if (frameIdx > 0) { frameIdx--; drawFrame(frameIdx); }
};
document.getElementById('btnNext').onclick = () => {
    if (frameIdx < FRAMES.length - 1) { frameIdx++; drawFrame(frameIdx); }
};

document.getElementById('scrubber').oninput = (e) => {
    frameIdx = parseInt(e.target.value);
    drawFrame(frameIdx);
};

document.getElementById('speedSlider').oninput = (e) => {
    speed = parseInt(e.target.value);
    document.getElementById('speedLabel').textContent = speed + ' fps';
};

// Keyboard controls
document.addEventListener('keydown', (e) => {
    if (e.key === ' ') { e.preventDefault(); document.getElementById('btnPlay').click(); }
    if (e.key === 'ArrowLeft') { document.getElementById('btnPrev').click(); }
    if (e.key === 'ArrowRight') { document.getElementById('btnNext').click(); }
});

// Init
document.getElementById('scrubber').max = FRAMES.length - 1;
drawFrame(0);
if (playing) { animId = requestAnimationFrame(tick); }
document.getElementById('btnPlay').innerHTML = '&#10074;&#10074; Pause';
</script>
</body>
</html>"""
