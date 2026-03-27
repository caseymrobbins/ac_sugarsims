"""
run_parallel.py
---------------
Launches architecture experiment runs as parallel subprocesses.
Each run is a separate Python process for genuine multi-core usage.

Usage:
    python run_parallel.py              # auto-detect cores
    python run_parallel.py --workers 6  # explicit
    python run_parallel.py --only C1_vanilla_sum  # single condition
"""

import subprocess
import sys
import os
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

SEEDS = [56, 663, 666, 1233, 5664, 223]
N_STEPS = 2000

# (name, objective, use_sevc, use_innovation, use_trust, trust_noise, use_hi)
CONDITIONS = [
    ("C1_vanilla_sum",      "SUM_RAW", False, True,  False, 0.0,  False),
    ("C2_sevc_sum",         "SUM_RAW", True,  True,  False, 0.0,  False),
    ("C3_sevc_hi_sum",      "SUM_RAW", True,  True,  False, 0.0,  True),
    ("C4_sevc_trust_sum",   "SUM_RAW", True,  True,  True,  0.1,  False),
    ("C5_sevc_trust_hi_sum","SUM_RAW", True,  True,  True,  0.1,  True),
    ("C6_full_topo",        "TOPO_X",  True,  True,  True,  0.1,  True),
]


SCRIPT_TEMPLATE = r'''
import sys, os, time
import numpy as np
import pandas as pd

os.chdir("@@CWD@@")
sys.path.insert(0, "@@CWD@@")

from environment import EconomicModel

# ── Patches ──

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

# ── Config (injected) ──

COND_NAME = "@@NAME@@"
OBJECTIVE = "@@OBJECTIVE@@"
USE_SEVC = @@USE_SEVC@@
USE_INNOVATION = @@USE_INNOVATION@@
USE_TRUST = @@USE_TRUST@@
TRUST_NOISE = @@TRUST_NOISE@@
USE_HI = @@USE_HI@@
SEED = @@SEED@@
N_STEPS = @@N_STEPS@@

# ── Run ──

model = EconomicModel(
    seed=SEED, grid_width=80, grid_height=80,
    n_workers=400, n_firms=20, n_landowners=15,
    objective=OBJECTIVE,
)

model.use_sevc = USE_SEVC
model.use_innovation = USE_INNOVATION
model.use_trust = USE_TRUST
model.trust_noise = TRUST_NOISE
model.use_horizon_index = USE_HI

if not USE_SEVC:
    for firm in model.firms:
        if hasattr(firm, 'strategy_weights'):
            for k in firm.strategy_weights:
                firm.strategy_weights[k] = 0.2

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

os.makedirs("results/architecture/raw_data", exist_ok=True)
df = pd.DataFrame(model.metrics_history)
df["condition"] = COND_NAME
df["objective"] = OBJECTIVE
df["use_sevc"] = USE_SEVC
df["use_innovation"] = USE_INNOVATION
df["use_trust"] = USE_TRUST
df["trust_noise"] = TRUST_NOISE
df["use_hi"] = USE_HI
df["seed"] = SEED
outpath = "results/architecture/raw_data/" + COND_NAME + "_seed" + str(SEED) + ".parquet"
df.to_parquet(outpath, index=False)
print("Saved: " + outpath)
'''


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def make_script(name, objective, use_sevc, use_innovation, use_trust, trust_noise, use_hi, seed, n_steps):
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
    s = s.replace("@@SEED@@", str(seed))
    s = s.replace("@@N_STEPS@@", str(n_steps))
    return s


def run_one(job):
    """Write script, run as subprocess, return result."""
    name, objective, use_sevc, use_innovation, use_trust, trust_noise, use_hi, seed, n_steps = job
    label = name + "/seed" + str(seed)

    script = make_script(name, objective, use_sevc, use_innovation,
                         use_trust, trust_noise, use_hi, seed, n_steps)

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
    parser.add_argument("--steps", type=int, default=N_STEPS)
    args = parser.parse_args()

    n_workers = args.workers
    if n_workers <= 0:
        import multiprocessing
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    jobs = []
    for cond in CONDITIONS:
        name = cond[0]
        if args.only and name != args.only:
            continue
        for seed in SEEDS:
            jobs.append(cond + (seed, args.steps))

    print("=" * 70)
    print("  PARALLEL ARCHITECTURE EXPERIMENT")
    print("  Jobs: " + str(len(jobs)))
    print("  Workers: " + str(n_workers))
    print("  Steps: " + str(args.steps))
    print("=" * 70)
    for cond in CONDITIONS:
        if args.only and cond[0] != args.only:
            continue
        name, obj, sevc, inno, trust, noise, hi = cond
        flags = []
        if sevc: flags.append("SEVC")
        if inno: flags.append("Inno")
        if trust: flags.append("Trust(" + str(noise) + ")")
        if hi: flags.append("HI")
        print("  " + name + ": " + obj + " [" + ", ".join(flags) + "]")
    print()

    os.makedirs("results/architecture/raw_data", exist_ok=True)

    t0 = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_one, job): job for job in jobs}
        results = []
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                job = futures[future]
                print("  EXCEPTION: " + str(job[0]) + "/" + str(job[-2]) + ": " + str(e))

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
    pd.DataFrame(results).to_csv("results/architecture/run_log.csv", index=False)


if __name__ == "__main__":
    main()
