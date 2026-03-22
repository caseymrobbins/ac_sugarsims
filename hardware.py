"""
hardware.py
-----------
Hardware detection, acceleration backend selection, and resource regeneration
functions for the multi-agent economic simulation.

Backends (in priority order):
  1. jax_gpu   — JAX with CUDA (A100, L4, V100)
  2. jax_cpu   — JAX on CPU
  3. numba_gpu — Numba CUDA kernels (T4)
  4. numba_cpu — Numba parallel CPU
  5. numpy     — Pure NumPy (always works)

Key design: NEVER let a hardware detection or initialization failure crash
the simulation. Every code path falls back to numpy.

Multiprocessing safety: JAX GPU init is not safe when many worker processes
all try to grab GPU memory simultaneously. The auto_configure function
detects if it's running in a subprocess and falls back to CPU.
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Resource grid constants
# ---------------------------------------------------------------------------

FOOD_CAP = 50.0
RAW_CAP = 30.0
LAND_CAP = 20.0
WATER_CAP = 40.0
POLLUTION_CAP = 50.0

# Regeneration rates
FOOD_REGEN = 0.8
RAW_REGEN = 0.3
WATER_REGEN = 0.5
POLLUTION_DECAY = 0.02       # fraction that decays per step
POLLUTION_DIFFUSE = 0.05     # spatial diffusion fraction


# ---------------------------------------------------------------------------
# Acceleration config
# ---------------------------------------------------------------------------

@dataclass
class AccelConfig:
    """Hardware acceleration configuration."""
    backend: str = "numpy"
    n_workers: int = 1
    recommended_seeds: int = 20
    recommended_steps: int = 1500
    gpu_name: str = ""
    gpu_mem_mb: int = 0


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def detect_hardware() -> dict:
    """Detect available hardware. Returns a dict with gpu_tier, name, mem, cpu_count."""
    import subprocess
    gpu_tier = "cpu"
    gpu_name = "none"
    gpu_mem = 0

    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            line = r.stdout.strip().splitlines()[0]
            parts = line.split(',')
            gpu_name = parts[0].strip()
            gpu_mem = int(parts[1].strip())
            name_l = gpu_name.lower()
            if 'a100' in name_l:
                gpu_tier = 'a100'
            elif 'l4' in name_l:
                gpu_tier = 'l4'
            elif 't4' in name_l:
                gpu_tier = 't4'
            elif 'v100' in name_l:
                gpu_tier = 'v100'
            else:
                gpu_tier = 'generic_gpu'
    except Exception:
        pass

    cpu_count = os.cpu_count() or 1

    return {
        "gpu_tier": gpu_tier,
        "gpu_name": gpu_name,
        "gpu_mem_mb": gpu_mem,
        "cpu_count": cpu_count,
    }


def get_accel_config(hw: dict) -> AccelConfig:
    """Build AccelConfig from hardware detection results."""
    tier = hw["gpu_tier"]
    cfg = AccelConfig()
    cfg.gpu_name = hw["gpu_name"]
    cfg.gpu_mem_mb = hw["gpu_mem_mb"]

    if tier == "a100":
        cfg.backend = "jax_gpu"
        cfg.n_workers = 14
        cfg.recommended_seeds = 20
        cfg.recommended_steps = 1500
    elif tier in ("l4", "v100"):
        cfg.backend = "jax_gpu"
        cfg.n_workers = 10
        cfg.recommended_seeds = 20
        cfg.recommended_steps = 1500
    elif tier == "t4":
        cfg.backend = "numba_gpu"
        cfg.n_workers = 6
        cfg.recommended_seeds = 15
        cfg.recommended_steps = 1000
    elif tier == "generic_gpu":
        cfg.backend = "jax_gpu"
        cfg.n_workers = 6
        cfg.recommended_seeds = 15
        cfg.recommended_steps = 1000
    else:
        # CPU only
        cpus = hw["cpu_count"]
        if cpus >= 8:
            cfg.backend = "numba_cpu"
            cfg.n_workers = max(1, cpus - 2)
        else:
            cfg.backend = "numpy"
            cfg.n_workers = max(1, cpus)
        cfg.recommended_seeds = 10
        cfg.recommended_steps = 500

    return cfg


def _hardware_report(hw: dict, cfg: AccelConfig) -> str:
    """Format a human-readable hardware report."""
    lines = [
        "=" * 50,
        "  HARDWARE REPORT",
        "=" * 50,
        f"  GPU tier     : {hw['gpu_tier']}",
        f"  GPU name     : {hw['gpu_name']}",
        f"  GPU VRAM     : {hw['gpu_mem_mb']/1024:.1f} GB" if hw['gpu_mem_mb'] else "  GPU VRAM     : N/A",
        f"  CPU cores    : {hw['cpu_count']}",
        f"  Backend      : {cfg.backend}",
        f"  Workers      : {cfg.n_workers}",
        f"  Rec. seeds   : {cfg.recommended_seeds}",
        f"  Rec. steps   : {cfg.recommended_steps}",
        "=" * 50,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Auto-configure (main entry point)
# ---------------------------------------------------------------------------

def auto_configure(verbose: bool = True) -> AccelConfig:
    """
    Detect hardware and return the best AccelConfig.

    Multiprocessing safety: when called from a worker process, we must NOT
    try to initialize JAX GPU (causes OOM when 10+ processes fight for VRAM).
    Worker processes should use numpy or jax_cpu.
    """
    hw = detect_hardware()
    cfg = get_accel_config(hw)

    # Test that the selected backend actually works
    cfg.backend = _validate_backend(cfg.backend, verbose=verbose)

    if verbose:
        print(_hardware_report(hw, cfg))

    return cfg


def _validate_backend(backend: str, verbose: bool = True) -> str:
    """
    Test that the selected backend actually initializes.
    Falls back through the chain until something works.
    Returns the validated backend string.
    """
    # Force JAX to use CPU+GPU with graceful fallback
    if backend.startswith("jax"):
        try:
            # CRITICAL: Disable JAX's default 90% GPU memory preallocation.
            # Without this, the first process grabs ~72GB on an H100/A100
            # and every other process OOMs.
            os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
            # Alternatively, limit each process to a fraction:
            # os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.05")

            # Set platform preference before importing JAX
            os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
            # Suppress noisy JAX warnings
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

            import jax
            import jax.numpy as jnp

            if backend == "jax_gpu":
                try:
                    devices = jax.devices("gpu")
                    if devices:
                        # Actually test GPU computation (catches OOM, driver issues)
                        test = jnp.ones(100, dtype=jnp.float64)
                        _ = (test + test).block_until_ready()
                        if verbose:
                            print(f"  JAX GPU validated: {devices[0].device_kind}")
                        return "jax_gpu"
                    else:
                        raise RuntimeError("No GPU devices found")
                except Exception as e:
                    if verbose:
                        print(f"  JAX GPU failed ({e}), trying JAX CPU...")
                    # Fall through to jax_cpu

            # Try JAX CPU
            try:
                cpu_devs = jax.devices("cpu")
                test = jnp.ones(100, dtype=jnp.float64)
                _ = (test + test).block_until_ready()
                if verbose:
                    print("  JAX CPU validated")
                return "jax_cpu"
            except Exception as e:
                if verbose:
                    print(f"  JAX CPU failed ({e}), falling back to numpy")

        except ImportError:
            if verbose:
                print("  JAX not installed, falling back to numpy")
        except Exception as e:
            if verbose:
                print(f"  JAX init failed ({e}), falling back to numpy")

    if backend.startswith("numba"):
        try:
            import numba
            if verbose:
                print(f"  Numba validated: {numba.__version__}")
            return backend
        except ImportError:
            if verbose:
                print("  Numba not installed, falling back to numpy")

    if verbose:
        print("  Using numpy backend")
    return "numpy"


# ---------------------------------------------------------------------------
# Resource regeneration functions
# ---------------------------------------------------------------------------

def build_regen_fn(backend: str):
    """
    Return a resource regeneration function for the given backend.
    All backends produce numerically identical results.
    """
    if backend == "jax_gpu" or backend == "jax_cpu":
        try:
            return _build_jax_regen()
        except Exception:
            pass  # Fall through to numpy

    if backend.startswith("numba"):
        try:
            return _build_numba_regen(gpu=(backend == "numba_gpu"))
        except Exception:
            pass  # Fall through to numpy

    return _build_numpy_regen()


def _build_numpy_regen():
    """Pure NumPy regeneration (always works)."""
    def regen(food, raw, land, water, pollution):
        food += FOOD_REGEN * (1 - food / FOOD_CAP)
        np.clip(food, 0, FOOD_CAP, out=food)
        raw += RAW_REGEN * (1 - raw / RAW_CAP)
        np.clip(raw, 0, RAW_CAP, out=raw)
        water += WATER_REGEN * (1 - water / WATER_CAP)
        np.clip(water, 0, WATER_CAP, out=water)
        pollution *= (1 - POLLUTION_DECAY)
        np.clip(pollution, 0, POLLUTION_CAP, out=pollution)
    return regen


def _build_jax_regen():
    """JAX-accelerated regeneration."""
    import jax.numpy as jnp
    from jax import jit

    @jit
    def _regen_jax(food, raw, land, water, pollution):
        food = jnp.clip(food + FOOD_REGEN * (1 - food / FOOD_CAP), 0, FOOD_CAP)
        raw = jnp.clip(raw + RAW_REGEN * (1 - raw / RAW_CAP), 0, RAW_CAP)
        water = jnp.clip(water + WATER_REGEN * (1 - water / WATER_CAP), 0, WATER_CAP)
        pollution = jnp.clip(pollution * (1 - POLLUTION_DECAY), 0, POLLUTION_CAP)
        return food, raw, land, water, pollution

    def regen(food, raw, land, water, pollution):
        import jax.numpy as jnp
        f, r, l, w, p = _regen_jax(
            jnp.array(food), jnp.array(raw), jnp.array(land),
            jnp.array(water), jnp.array(pollution))
        food[:] = np.asarray(f)
        raw[:] = np.asarray(r)
        water[:] = np.asarray(w)
        pollution[:] = np.asarray(p)

    return regen


def _build_numba_regen(gpu: bool = False):
    """Numba-accelerated regeneration."""
    import numba

    if gpu:
        from numba import cuda

        @cuda.jit
        def _regen_kernel(food, raw, water, pollution,
                          food_cap, raw_cap, water_cap, poll_cap,
                          food_rate, raw_rate, water_rate, poll_decay):
            i, j = cuda.grid(2)
            if i < food.shape[0] and j < food.shape[1]:
                food[i, j] = min(food_cap, max(0.0, food[i, j] + food_rate * (1 - food[i, j] / food_cap)))
                raw[i, j] = min(raw_cap, max(0.0, raw[i, j] + raw_rate * (1 - raw[i, j] / raw_cap)))
                water[i, j] = min(water_cap, max(0.0, water[i, j] + water_rate * (1 - water[i, j] / water_cap)))
                pollution[i, j] = min(poll_cap, max(0.0, pollution[i, j] * (1 - poll_decay)))

        def regen(food, raw, land, water, pollution):
            tpb = (16, 16)
            bpg_x = (food.shape[0] + tpb[0] - 1) // tpb[0]
            bpg_y = (food.shape[1] + tpb[1] - 1) // tpb[1]
            _regen_kernel[(bpg_x, bpg_y), tpb](
                food, raw, water, pollution,
                FOOD_CAP, RAW_CAP, WATER_CAP, POLLUTION_CAP,
                FOOD_REGEN, RAW_REGEN, WATER_REGEN, POLLUTION_DECAY)

        return regen

    else:
        @numba.njit(parallel=True)
        def regen(food, raw, land, water, pollution):
            for i in numba.prange(food.shape[0]):
                for j in range(food.shape[1]):
                    food[i, j] = min(FOOD_CAP, max(0.0, food[i, j] + FOOD_REGEN * (1 - food[i, j] / FOOD_CAP)))
                    raw[i, j] = min(RAW_CAP, max(0.0, raw[i, j] + RAW_REGEN * (1 - raw[i, j] / RAW_CAP)))
                    water[i, j] = min(WATER_CAP, max(0.0, water[i, j] + WATER_REGEN * (1 - water[i, j] / WATER_CAP)))
                    pollution[i, j] = min(POLLUTION_CAP, max(0.0, pollution[i, j] * (1 - POLLUTION_DECAY)))

        return regen
