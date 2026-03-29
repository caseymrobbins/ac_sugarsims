"""
run_animations.py
-----------------
Generate HTML5 animations for all 8 experiment conditions.

Runs each condition for a configurable number of steps (default 500)
with animation frame collection enabled, then generates interactive
HTML players in results/animations/.

Usage:
    python run_animations.py                  # 500 steps, all conditions
    python run_animations.py --steps 1000     # more steps
    python run_animations.py --only C4_full_auth  # single condition
    python run_animations.py --subsample 3    # every 3rd frame (smaller files)
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import traceback

from run_architecture_experiment import CONDITIONS, Condition, configure_model, apply_patches

SEED = 42
GRID_SIZE = 80
N_WORKERS = 400
N_FIRMS = 20
N_LANDOWNERS = 15
OUTPUT_DIR = "results/animations"


def run_and_animate(condition: Condition, n_steps: int, subsample: int) -> str:
    """Run one condition with animation collection and generate HTML."""
    from environment import EconomicModel
    from trust import update_trust_scores
    from animate import generate_animation_html

    label = condition.name
    print(f"  [{label}] {condition.label}", flush=True)
    print(f"    gov={condition.gov_type}, sevc={condition.use_sevc}, "
          f"firm_hi={condition.use_firm_hi}, mix={condition.mixed_sevc_ratio}")
    t0 = time.time()

    model = EconomicModel(
        seed=SEED,
        grid_width=GRID_SIZE,
        grid_height=GRID_SIZE,
        n_workers=N_WORKERS,
        n_firms=N_FIRMS,
        n_landowners=N_LANDOWNERS,
        objective=condition.objective,
    )
    model._collect_animation = True
    configure_model(model, condition)

    for step in range(n_steps):
        model.step()
        if condition.use_trust:
            update_trust_scores(model)
        if (step + 1) % 100 == 0:
            print(f"    step {step + 1}/{n_steps}", flush=True)

    elapsed = time.time() - t0
    n_frames = len(model.animation_frames)
    print(f"    simulation done: {elapsed:.1f}s, {n_frames} frames")

    # Generate HTML animation
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = f"{OUTPUT_DIR}/{label}.html"

    title = f"{label}: {condition.label} (gov={condition.gov_type})"
    generate_animation_html(
        model.animation_frames,
        output_path=out_path,
        grid_size=GRID_SIZE,
        title=title,
        subsample=subsample,
    )

    fsize_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"    saved: {out_path} ({fsize_mb:.1f} MB)")
    return out_path


def generate_index_page(paths: list, n_steps: int):
    """Generate an index.html linking to all animation files."""
    rows = ""
    for cond, path in paths:
        fname = os.path.basename(path)
        rows += (f'<tr><td><a href="{fname}">{cond.name}</a></td>'
                 f'<td>{cond.label}</td><td>{cond.gov_type}</td>'
                 f'<td>{"Yes" if cond.use_sevc else "No"}</td>'
                 f'<td>{"Yes" if cond.use_firm_hi else "No"}</td>'
                 f'<td>{cond.mixed_sevc_ratio}</td></tr>\n')

    html = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Simulation Animations Index</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
th {{ background: #2c3e50; color: white; }}
tr:nth-child(even) {{ background: #f9f9f9; }}
a {{ color: #2980b9; }}
h1 {{ color: #2c3e50; }}
</style></head><body>
<h1>Simulation Animations</h1>
<p>{len(paths)} conditions, {n_steps} steps each, seed={SEED}</p>
<table>
<tr><th>Condition</th><th>Label</th><th>Gov Type</th><th>SEVC</th><th>Firm HI</th><th>Mix Ratio</th></tr>
{rows}</table>
</body></html>"""

    idx_path = f"{OUTPUT_DIR}/index.html"
    with open(idx_path, "w") as f:
        f.write(html)
    print(f"\nIndex page: {idx_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate animations for all experiment conditions")
    parser.add_argument("--steps", type=int, default=500, help="Steps per simulation (default: 500)")
    parser.add_argument("--only", type=str, default=None, help="Run only this condition name")
    parser.add_argument("--subsample", type=int, default=2, help="Use every Nth frame (default: 2)")
    args = parser.parse_args()

    print("=" * 60)
    print("  ANIMATION GENERATOR")
    print(f"  Steps: {args.steps}")
    print(f"  Seed: {SEED}")
    print(f"  Subsample: every {args.subsample} frames")
    print(f"  Output: {OUTPUT_DIR}/")
    print("=" * 60)

    # Apply feature patches once
    print("\nApplying patches...")
    apply_patches()
    print()

    # Filter conditions
    conditions = CONDITIONS
    if args.only:
        conditions = [c for c in CONDITIONS if c.name == args.only]
        if not conditions:
            print(f"ERROR: condition '{args.only}' not found")
            print(f"Available: {', '.join(c.name for c in CONDITIONS)}")
            sys.exit(1)

    total_t0 = time.time()
    results = []

    for cond in conditions:
        try:
            path = run_and_animate(cond, args.steps, args.subsample)
            results.append((cond, path))
        except Exception as e:
            print(f"  FAIL: {cond.name}: {e}")
            traceback.print_exc()
        print()

    total_elapsed = time.time() - total_t0

    print("=" * 60)
    print(f"  Generated {len(results)}/{len(conditions)} animations in {total_elapsed:.1f}s")
    print("=" * 60)

    if results:
        generate_index_page(results, args.steps)
        print(f"\nOpen {OUTPUT_DIR}/index.html to browse all animations")


if __name__ == "__main__":
    main()
