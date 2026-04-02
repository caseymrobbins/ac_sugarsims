"""
run_parallel.py
---------------
Launches architecture experiment runs as parallel subprocesses.
Each run is a separate Python process for genuine multi-core usage.

11 conditions testing SEVC layers, governance types, mixed populations,
and planner SEVC objectives:
  C1-C3: Feature staircase (baseline -> SEVC -> HI)
  C4-C7: Governance comparison (auth, demo_captured, auth_captured, democratic)
  C8: Mixed 50/50 SEVC/Vanilla competition
  C9-C11: Planner SEVC objective (democratic, authoritarian, captured demo)

Usage:
    python run_parallel.py              # auto-detect cores
    python run_parallel.py --workers 6  # explicit
    python run_parallel.py --only C1_baseline  # single condition
    python run_parallel.py --animate    # generate HTML animations for each run
    python run_parallel.py --preset test2  # run 2-condition test with 10 seeds
"""

import subprocess
import sys
import os
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

SEEDS = [42, 101, 137, 202, 256, 303, 389, 404, 501, 505, 606, 623, 777, 888, 999]
N_STEPS = 2000

# (name, objective, use_sevc, use_innovation, use_trust, trust_noise, use_hi, use_firm_hi, gov_type,
#   mixed_sevc_ratio, election_weight, media_captured, production_aware_E, production_aware_S_pop,
#   ceo_compensation_tied, ceo_base_equals_floor, ceo_equity_tied, capture_normalization)
CONDITIONS = [
    # ("C16_production_aware_democratic",  "PLANNER_SEVC", True,  True, True, 0.1, True, True, "democratic",    1.0, 2.0, False, True,  True,  False, False, False, "fixed"),
    # ("C17_production_aware_no_sevc",     "PLANNER_SEVC", False, True, True, 0.1, True, True, "democratic",    1.0, 2.0, False, False, True,  False, False, False, "fixed"),
    # ("C18_production_aware_captured",    "PLANNER_SEVC", True,  True, True, 0.1, True, True, "demo_captured", 1.0, 2.0, True,  True,  True,  False, False, False, "fixed"),
    # Task 12: CEO compensation tied to SEVC floor
    ("C19_ceo_tied_democratic",          "PLANNER_SEVC", True,  True, True, 0.1, True, True, "democratic",    1.0, 2.0, False, True,  True,  True,  True,  True,  "ema"),
    ("C20_ceo_tied_captured",            "PLANNER_SEVC", True,  True, True, 0.1, True, True, "demo_captured", 1.0, 2.0, True,  True,  True,  True,  True,  True,  "ema"),
]

# Preset: 2-condition test (vanilla vs full stack) with 10 seeds
TEST2_CONDITIONS = [
    ("vanilla_sum",   "SUM_RAW",      False, True, False, 0.0, False, False, "authoritarian", 1.0, 0.0, False),
    ("topo_sevc_hi",  "TOPO_X",       True,  True, True,  0.1, True,  True,  "democratic",    1.0, 0.0, False),
]
TEST2_SEEDS = [7, 23, 59, 101, 233, 347, 461, 587, 719, 853]

# Task 11: Production-Aware Capital conditions (C16/C17/C18)
# Tuple: (name, objective, use_sevc, use_innovation, use_trust, trust_noise,
#          use_hi, use_firm_hi, gov_type, mixed_sevc_ratio, election_weight,
#          media_captured, production_aware_E, production_aware_S_pop)
PA_CONDITIONS = [
    # C16: full production-aware SEVC, responsive democracy — primary test
    ("C16_production_aware_democratic",  "PLANNER_SEVC", True,  True, True, 0.1, True, True, "democratic",    1.0, 2.0, False, True,  True),
    # C17: vanilla firms + planner capture floor — planner-only baseline
    ("C17_production_aware_no_sevc",     "PLANNER_SEVC", False, True, True, 0.1, True, True, "democratic",    1.0, 2.0, False, False, True),
    # C18: production-aware SEVC + captured media — robustness test
    ("C18_production_aware_captured",    "PLANNER_SEVC", True,  True, True, 0.1, True, True, "demo_captured", 1.0, 2.0, True,  True,  True),
]
PA_SEEDS  = [42, 137, 256, 389, 501, 623, 777, 888]
PA_STEPS  = 2000

# Preset: C12-C15 responsiveness test with 8 new seeds
RESP_CONDITIONS = [
    ("C12_responsive_democratic",     "PLANNER_SEVC", True, True, True, 0.1, True, True, "democratic",    1.0, 2.0, False),
    ("C13_responsive_demo_captured",  "PLANNER_SEVC", True, True, True, 0.1, True, True, "demo_captured", 1.0, 2.0, True),
    ("C14_pure_technocrat_democratic", "PLANNER_SEVC", True, True, True, 0.1, True, True, "democratic",    1.0, 0.0, False),
    ("C15_pure_technocrat_auth",       "PLANNER_SEVC", True, True, True, 0.1, True, True, "authoritarian", 1.0, 0.0, False),
]
RESP_SEEDS = [42, 137, 256, 389, 501, 623, 777, 888]


SCRIPT_TEMPLATE = r'''
import sys, os, time
import numpy as np
import pandas as pd

os.chdir("@@CWD@@")
sys.path.insert(0, "@@CWD@@")

from environment import EconomicModel

# -- Patches --

import sustainable_capitalism as sc_module
_oc = sc_module.sustainable_choose_strategy
_ol = sc_module.sustainable_learn_from_outcome

def pc(firm):
    if not getattr(firm.model, 'use_sevc', True):
        strats = list(firm.strategy_weights.keys())
        return strats[firm.model.rng.integers(len(strats))] if strats else "expand"
    return _oc(firm)

def pl(firm, strategy, reward):
    if not getattr(firm.model, 'use_sevc', True):
        return
    return _ol(firm, strategy, reward)

sc_module.sustainable_choose_strategy = pc
sc_module.sustainable_learn_from_outcome = pl

import innovation as inno
_ord = inno.firm_rd_invest
_odf = inno.diffuse_technology

def prd(firm):
    if not getattr(firm.model, 'use_innovation', True):
        return
    return _ord(firm)

def pdf(model):
    if not getattr(model, 'use_innovation', True):
        return
    return _odf(model)

inno.firm_rd_invest = prd
inno.diffuse_technology = pdf

import trust as trust_mod
_ot = trust_mod.update_trust_scores

def pt(model):
    if getattr(model, '_trust_frozen', False):
        return
    _ot(model)
    noise = getattr(model, 'trust_noise', 0.0)
    if noise > 0:
        rng = model.rng
        for agent in (model.schedule.agents if hasattr(model, 'schedule') else []):
            if hasattr(agent, 'trust_score'):
                agent.trust_score = float(np.clip(
                    agent.trust_score + rng.normal(0, noise), 0.0, 1.0))

trust_mod.update_trust_scores = pt

import planner as planner_mod
if hasattr(planner_mod, '_get_horizon_index'):
    _ohi = planner_mod._get_horizon_index
    def phi(model):
        if not getattr(model, 'use_horizon_index', True):
            return 1.0
        return _ohi(model)
    planner_mod._get_horizon_index = phi

# -- Config (injected) --

COND_NAME = "@@NAME@@"
OBJECTIVE = "@@OBJECTIVE@@"
USE_SEVC = @@USE_SEVC@@
USE_INNOVATION = @@USE_INNOVATION@@
USE_TRUST = @@USE_TRUST@@
TRUST_NOISE = @@TRUST_NOISE@@
USE_HI = @@USE_HI@@
USE_FIRM_HI = @@USE_FIRM_HI@@
GOV_TYPE = "@@GOV_TYPE@@"
MIXED_SEVC_RATIO = @@MIXED_SEVC_RATIO@@
ELECTION_WEIGHT = @@ELECTION_WEIGHT@@
MEDIA_CAPTURED = @@MEDIA_CAPTURED@@
PRODUCTION_AWARE_E    = @@PRODUCTION_AWARE_E@@
PRODUCTION_AWARE_S_POP = @@PRODUCTION_AWARE_S_POP@@
CEO_COMPENSATION_TIED = @@CEO_COMPENSATION_TIED@@
CEO_BASE_EQUALS_FLOOR = @@CEO_BASE_EQUALS_FLOOR@@
CEO_EQUITY_TIED       = @@CEO_EQUITY_TIED@@
CAPTURE_NORMALIZATION = "@@CAPTURE_NORMALIZATION@@"
SEED = @@SEED@@
N_STEPS = @@N_STEPS@@
ANIMATE = @@ANIMATE@@
ANIM_SUBSAMPLE = @@ANIM_SUBSAMPLE@@
OUTPUT_DIR = "@@OUTPUT_DIR@@"

# -- Run --

model = EconomicModel(
    seed=SEED, grid_width=80, grid_height=80,
    n_workers=400, n_firms=20, n_landowners=15,
    objective=OBJECTIVE,
)

model._collect_animation = ANIMATE
model.use_sevc = USE_SEVC
model.use_innovation = USE_INNOVATION
model.use_trust = USE_TRUST
model.trust_noise = TRUST_NOISE
model.use_horizon_index = USE_HI
model.use_firm_hi = USE_FIRM_HI
model.gov_type = GOV_TYPE
model.election_weight = ELECTION_WEIGHT
model.production_aware_E     = PRODUCTION_AWARE_E
model.production_aware_S_pop = PRODUCTION_AWARE_S_POP
model.ceo_compensation_tied  = CEO_COMPENSATION_TIED
model.ceo_base_equals_floor  = CEO_BASE_EQUALS_FLOOR
model.ceo_equity_tied        = CEO_EQUITY_TIED
model.capture_normalization  = CAPTURE_NORMALIZATION

if not USE_SEVC:
    for firm in model.firms:
        firm.is_sevc = False
        if hasattr(firm, 'strategy_weights'):
            for k in firm.strategy_weights:
                firm.strategy_weights[k] = 0.2
else:
    from collections import deque as _dq
    for firm in model.firms:
        if not hasattr(firm, 'floor_history'):
            firm.floor_history = _dq(maxlen=100)
            firm.horizon_index = 1.0

if MIXED_SEVC_RATIO < 1.0 and USE_SEVC:
    n_vanilla = int(len(model.firms) * (1.0 - MIXED_SEVC_RATIO))
    if n_vanilla > 0:
        indices = list(range(len(model.firms)))
        model.rng.shuffle(indices)
        for i in indices[:n_vanilla]:
            firm = model.firms[i]
            firm.is_sevc = False
            if hasattr(firm, 'strategy_weights'):
                for k in firm.strategy_weights:
                    firm.strategy_weights[k] = 0.2

if (GOV_TYPE == "demo_captured" or MEDIA_CAPTURED) and hasattr(model, 'news_firms'):
    for nf in model.news_firms:
        if hasattr(nf, 'accuracy') and nf.accuracy > 0.5:
            nf.accuracy = 0.3; nf.audience_capture = 0.6
            break

if not USE_TRUST:
    model._trust_frozen = True
    for agent in (model.schedule.agents if hasattr(model, 'schedule') else []):
        if hasattr(agent, 'trust_score'):
            agent.trust_score = 0.5
else:
    model._trust_frozen = False

t0 = time.time()
for step in range(N_STEPS):
    model.step()
    if (step + 1) % 500 == 0:
        elapsed_so_far = time.time() - t0
        print(str(step + 1) + " (" + "{:.0f}".format(elapsed_so_far) + "s)", end=" ", flush=True)

elapsed = time.time() - t0
print("done " + "{:.1f}".format(elapsed) + "s")

os.makedirs(OUTPUT_DIR + "/raw_data", exist_ok=True)
df = pd.DataFrame(model.metrics_history)
df["condition"] = COND_NAME
df["objective"] = OBJECTIVE
df["use_sevc"] = USE_SEVC
df["use_innovation"] = USE_INNOVATION
df["use_trust"] = USE_TRUST
df["trust_noise"] = TRUST_NOISE
df["use_hi"] = USE_HI
df["use_firm_hi"] = USE_FIRM_HI
df["gov_type"] = GOV_TYPE
df["seed"] = SEED
outpath = OUTPUT_DIR + "/raw_data/" + COND_NAME + "_seed" + str(SEED) + ".parquet"
df.to_parquet(outpath, index=False)
print("Saved: " + outpath)

if ANIMATE and model.animation_frames:
    anim_dir = OUTPUT_DIR + "/animations"
    os.makedirs(anim_dir, exist_ok=True)
    try:
        from animate import generate_animation_html
        anim_path = anim_dir + "/" + COND_NAME + "_seed" + str(SEED) + ".html"
        generate_animation_html(
            model.animation_frames,
            output_path=anim_path,
            grid_size=80,
            title=COND_NAME + " (seed=" + str(SEED) + ")",
            subsample=ANIM_SUBSAMPLE,
        )
        fsize = os.path.getsize(anim_path) / (1024 * 1024)
        print("Animation: " + anim_path + " (" + "{:.1f}".format(fsize) + " MB)")
    except Exception as e:
        print("Animation failed: " + str(e))
'''


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def make_script(name, objective, use_sevc, use_innovation, use_trust, trust_noise,
                use_hi, use_firm_hi, gov_type, mixed_sevc_ratio,
                election_weight, media_captured,
                seed, n_steps,
                animate=False, anim_subsample=2, output_dir="results/architecture",
                production_aware_E=False, production_aware_S_pop=False,
                ceo_compensation_tied=False, ceo_base_equals_floor=False,
                ceo_equity_tied=False, capture_normalization="fixed"):
    """Generate a self-contained run script with config injected."""
    s = SCRIPT_TEMPLATE
    s = s.replace("@@CWD@@", _SCRIPT_DIR)
    s = s.replace("@@NAME@@", name)
    s = s.replace("@@OBJECTIVE@@", objective)
    s = s.replace("@@USE_SEVC@@", str(use_sevc))
    s = s.replace("@@USE_INNOVATION@@", str(use_innovation))
    s = s.replace("@@USE_TRUST@@", str(use_trust))
    s = s.replace("@@TRUST_NOISE@@", str(trust_noise))
    s = s.replace("@@USE_HI@@", str(use_hi))
    s = s.replace("@@USE_FIRM_HI@@", str(use_firm_hi))
    s = s.replace("@@GOV_TYPE@@", str(gov_type))
    s = s.replace("@@MIXED_SEVC_RATIO@@", str(mixed_sevc_ratio))
    s = s.replace("@@ELECTION_WEIGHT@@", str(election_weight))
    s = s.replace("@@MEDIA_CAPTURED@@", str(media_captured))
    s = s.replace("@@PRODUCTION_AWARE_E@@", str(production_aware_E))
    s = s.replace("@@PRODUCTION_AWARE_S_POP@@", str(production_aware_S_pop))
    s = s.replace("@@CEO_COMPENSATION_TIED@@", str(ceo_compensation_tied))
    s = s.replace("@@CEO_BASE_EQUALS_FLOOR@@", str(ceo_base_equals_floor))
    s = s.replace("@@CEO_EQUITY_TIED@@", str(ceo_equity_tied))
    s = s.replace("@@CAPTURE_NORMALIZATION@@", str(capture_normalization))
    s = s.replace("@@SEED@@", str(seed))
    s = s.replace("@@N_STEPS@@", str(n_steps))
    s = s.replace("@@ANIMATE@@", str(animate))
    s = s.replace("@@ANIM_SUBSAMPLE@@", str(anim_subsample))
    s = s.replace("@@OUTPUT_DIR@@", output_dir)
    return s


def run_one(job):
    """Write script, run as subprocess, return result."""
    (name, objective, use_sevc, use_innovation, use_trust, trust_noise,
     use_hi, use_firm_hi, gov_type, mixed_sevc_ratio,
     election_weight, media_captured, production_aware_E, production_aware_S_pop,
     ceo_compensation_tied, ceo_base_equals_floor, ceo_equity_tied, capture_normalization,
     seed, n_steps, animate, anim_subsample, output_dir) = job
    label = name + "/seed" + str(seed)

    script = make_script(name, objective, use_sevc, use_innovation,
                         use_trust, trust_noise, use_hi, use_firm_hi, gov_type,
                         mixed_sevc_ratio, election_weight, media_captured,
                         seed, n_steps,
                         animate=animate, anim_subsample=anim_subsample,
                         output_dir=output_dir,
                         production_aware_E=production_aware_E,
                         production_aware_S_pop=production_aware_S_pop,
                         ceo_compensation_tied=ceo_compensation_tied,
                         ceo_base_equals_floor=ceo_base_equals_floor,
                         ceo_equity_tied=ceo_equity_tied,
                         capture_normalization=capture_normalization)

    script_path = "/tmp/run_" + name + "_s" + str(seed) + ".py"
    with open(script_path, "w") as f:
        f.write(script)

    print("  START: " + label, flush=True)
    t0 = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=7200,
            cwd=os.getcwd(),
        )
        elapsed = time.time() - t0

        if result.returncode != 0:
            print("  FAIL:  " + label + " (" + str(int(elapsed)) + "s)")
            err_lines = result.stderr.strip().split("\n")
            for line in err_lines[-5:]:
                print("    " + line)
            return {"name": name, "seed": seed, "status": "FAIL",
                    "elapsed": elapsed, "error": result.stderr[-300:]}
        else:
            print("  DONE:  " + label + " (" + str(int(elapsed)) + "s)")
            return {"name": name, "seed": seed, "status": "OK",
                    "elapsed": elapsed}
    except subprocess.TimeoutExpired:
        print("  TIMEOUT: " + label)
        return {"name": name, "seed": seed, "status": "TIMEOUT", "elapsed": 7200}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers (0 = auto)")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only this condition")
    parser.add_argument("--steps", type=int, default=0,
                        help="Steps per run (0 = use preset default)")
    parser.add_argument("--animate", action="store_true",
                        help="Generate HTML animation for each run")
    parser.add_argument("--subsample", type=int, default=2,
                        help="Animation frame subsample rate (default: 2)")
    parser.add_argument("--preset", type=str, default=None,
                        choices=["full", "test2", "resp"],
                        help="Preset: 'full' = all 15 conditions, 'test2' = vanilla vs topo (10 seeds), 'resp' = C12-C15 (8 seeds), 'pa' = Task 11 C16-C18 (8 seeds)")
    args = parser.parse_args()

    # Select conditions and seeds based on preset
    if args.preset == "test2":
        conditions = TEST2_CONDITIONS
        seeds = TEST2_SEEDS
        n_steps = args.steps if args.steps > 0 else 500
        output_dir = "results/test_conditions"
    elif args.preset == "resp":
        conditions = RESP_CONDITIONS
        seeds = RESP_SEEDS
        n_steps = args.steps if args.steps > 0 else N_STEPS
        output_dir = "results/responsiveness"
    elif args.preset == "pa":
        conditions = PA_CONDITIONS
        seeds = PA_SEEDS
        n_steps = args.steps if args.steps > 0 else PA_STEPS
        output_dir = "results/production_aware"
    else:
        conditions = CONDITIONS
        seeds = SEEDS
        n_steps = args.steps if args.steps > 0 else N_STEPS
        output_dir = "results/architecture"

    n_workers = args.workers
    if n_workers <= 0:
        import multiprocessing
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    jobs = []
    for cond in conditions:
        name = cond[0]
        if args.only and name != args.only:
            continue
        for seed in seeds:
            # Pad legacy tuples up to 18 fields
            full_cond = cond
            if len(full_cond) < 14:
                full_cond = full_cond + (False, False)        # pad production_aware flags
            if len(full_cond) < 18:
                full_cond = full_cond + (False, False, False, "fixed")  # pad CEO flags
            jobs.append(full_cond + (seed, n_steps, args.animate, args.subsample, output_dir))

    print("=" * 70)
    print("  PARALLEL ARCHITECTURE EXPERIMENT")
    if args.preset:
        print("  Preset: " + args.preset)
    print("  Jobs: " + str(len(jobs)))
    print("  Workers: " + str(n_workers))
    print("  Steps: " + str(n_steps))
    print("  Animate: " + str(args.animate))
    print("  Output: " + output_dir)
    print("=" * 70)
    for cond in conditions:
        if args.only and cond[0] != args.only:
            continue
        name, obj, sevc, inno, trust, noise, hi, firm_hi, gov, mixed_ratio, elec_w, media_cap = cond[:12]
        pa_e   = cond[12] if len(cond) > 12 else False
        pa_s   = cond[13] if len(cond) > 13 else False
        ceo_t  = cond[14] if len(cond) > 14 else False
        cap_n  = cond[17] if len(cond) > 17 else "fixed"
        flags = []
        if sevc: flags.append("SEVC")
        if inno: flags.append("Inno")
        if trust: flags.append("Trust(" + str(noise) + ")")
        if hi: flags.append("HI")
        if firm_hi: flags.append("FirmHI")
        flags.append("gov=" + gov)
        if mixed_ratio < 1.0: flags.append("Mix(" + str(mixed_ratio) + ")")
        if elec_w > 0: flags.append("Resp(" + str(elec_w) + ")")
        if media_cap: flags.append("MediaCap")
        if pa_e or pa_s: flags.append("PA(E=" + str(pa_e) + ",S=" + str(pa_s) + ")")
        if ceo_t: flags.append("CEO(norm=" + str(cap_n) + ")")
        print("  " + name + ": " + obj + " [" + ", ".join(flags) + "]")
    print()

    os.makedirs(output_dir + "/raw_data", exist_ok=True)

    t0 = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_one, job): job for job in jobs}
        results = []
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                job = futures[future]
                print("  EXCEPTION: " + str(job[0]) + "/" + str(job[-4]) + ": " + str(e))

    elapsed = time.time() - t0
    ok = sum(1 for r in results if r.get("status") == "OK")
    fail = len(results) - ok

    print()
    print("=" * 70)
    print("  Complete: " + str(len(results)) + " runs in " +
          str(int(elapsed)) + "s (" + "{:.1f}".format(elapsed/60) + " min)")
    print("  OK: " + str(ok) + "  FAIL: " + str(fail))
    print("=" * 70)

    import pandas as pd
    pd.DataFrame(results).to_csv(output_dir + "/run_log.csv", index=False)

    # Print comparison summary if test2 preset
    if args.preset == "test2":
        _print_comparison(output_dir)


def _print_comparison(output_dir):
    """Load raw data and print head-to-head comparison."""
    import pandas as pd
    raw_dir = output_dir + "/raw_data"
    files = sorted(f for f in os.listdir(raw_dir) if f.endswith(".parquet"))
    if not files:
        return
    all_data = pd.concat([pd.read_parquet(raw_dir + "/" + f) for f in files], ignore_index=True)

    max_step = all_data["step"].max()
    tail = all_data[all_data["step"] >= max_step * 0.8]
    conditions = sorted(tail["condition"].unique())

    metrics = [
        ("worker_min",            "Floor Wealth",     ".1f",  "higher"),
        ("worker_mean",           "Worker Mean",      ".1f",  "higher"),
        ("worker_gini",           "Worker Gini",      ".3f",  "lower"),
        ("unemployment_rate",     "Unemployment",     ".1%",  "lower"),
        ("agency_floor",          "Agency Floor",     ".2f",  "higher"),
        ("horizon_index",         "Horizon Index",    ".3f",  "higher"),
        ("mean_firm_floor",       "Firm SEVC Floor",  ".3f",  "higher"),
        ("mean_firm_floor_raw",   "Firm Floor Raw",   ".3f",  "higher"),
        ("mean_firm_floor_norm",  "Firm Floor Norm",  ".3f",  "higher"),
        ("mean_firm_hi",          "Firm HI",          ".3f",  "higher"),
        ("n_firms",               "Firms",            ".1f",  "higher"),
        ("total_production",      "Production",       ".0f",  "higher"),
        ("total_pollution",       "Pollution",        ".0f",  "lower"),
        ("legitimacy_mean",       "Legitimacy",       ".3f",  "higher"),
        ("mean_aggression",       "Aggression",       ".3f",  "lower"),
        ("mean_conflict",         "Conflict",         ".4f",  "lower"),
        ("planner_sevc_score",    "Planner SEVC",     ".3f",  "higher"),
        ("planner_S_pop",         "Planner S",        ".3f",  "higher"),
        ("planner_E_pop",         "Planner E",        ".3f",  "higher"),
        ("planner_V_pop",         "Planner V",        ".3f",  "higher"),
        ("planner_C_pop",         "Planner C",        ".3f",  "higher"),
        ("mean_firm_binding_S",   "Binding S frac",   ".2f",  "info"),
        ("mean_firm_binding_E",   "Binding E frac",   ".2f",  "info"),
        ("mean_firm_binding_V",   "Binding V frac",   ".2f",  "info"),
        ("mean_firm_binding_C",   "Binding C frac",   ".2f",  "info"),
        ("election_planner_aligned", "Elec-Plan Align", ".2f",  "higher"),
        ("trust_planner",         "Planner Trust",    ".3f",  "higher"),
        ("trust_institutional",   "Inst. Trust",      ".3f",  "higher"),
        ("crime_events",          "Crime/step",       ".1f",  "lower"),
    ]

    import numpy as np
    print("\n" + "=" * 80)
    print("  HEAD-TO-HEAD COMPARISON (steady-state mean +/- std across seeds)")
    print("=" * 80)

    header = "{:<20}".format("Metric")
    for c in conditions:
        header += "  {:>26}".format(c)
    print(header)
    print("-" * len(header))

    for key, label, fmt, direction in metrics:
        if key not in tail.columns:
            continue
        row = "{:<20}".format(label)
        for c in conditions:
            subset = tail[tail["condition"] == c][key]
            mean_v = subset.mean(); std_v = subset.std()
            row += "  {:>12{fmt}} +/- {:>7{fmt}}".format(mean_v, std_v, fmt=fmt)
        print(row)

    print()
    summary_dir = output_dir + "/summary"
    os.makedirs(summary_dir, exist_ok=True)
    all_data.to_parquet(summary_dir + "/all_data.parquet", index=False)
    print("Saved: " + summary_dir + "/all_data.parquet")


if __name__ == "__main__":
    main()
