"""
hardware.py — hardware detection and acceleration config.

Regen function signature: regen_fn(food, raw_mat, land, water, pollution) → None (in-place)
All arrays shape (W, H), dtype float64.

Agriculture bonus (from planner budget) is applied in environment.step()
by multiplying the regen rates before calling this function.

Pollution spatial diffusion is handled in environment.step() using NumPy rolls
(independent of backend) so the kernels here only apply the per-cell decay.
"""

from __future__ import annotations

import multiprocessing
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Optional


# ── Resource caps ────────────────────────────────────────────────────────────

FOOD_CAP,  FOOD_RATE  = 30.0, 0.15
RAW_CAP,   RAW_RATE   = 25.0, 0.05
LAND_CAP,  LAND_RATE  = 20.0, 0.01
WATER_CAP, WATER_RATE = 40.0, 0.12
NUDGE = 0.01

# Pollution: cap and natural decay rate (per step).
# Spatial diffusion is applied separately in environment.step().
POLLUTION_CAP   = 50.0
POLLUTION_DECAY = 0.02    # 2 % natural degradation per cell per step
POLLUTION_DIFFUSE = 0.08  # fraction that spreads to 4 neighbours per step


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class GPUInfo:
    name: str
    memory_mb: int
    compute_cap: str = ""

    @property
    def memory_gb(self): return self.memory_mb / 1024

    @property
    def is_a100(self): return "a100" in self.name.lower()
    @property
    def is_l4(self):   return "l4" in self.name.lower()
    @property
    def is_t4(self):   return "t4" in self.name.lower()
    @property
    def is_v100(self): return "v100" in self.name.lower()

    @property
    def tier(self):
        for t in ("a100", "l4", "t4", "v100"):
            if t in self.name.lower():
                return t
        return "generic"


@dataclass
class HardwareInfo:
    cpu_cores: int
    ram_gb: float
    gpus: List[GPUInfo] = field(default_factory=list)
    is_colab: bool = False
    has_jax: bool = False
    has_jax_gpu: bool = False
    has_numba: bool = False
    has_numba_cuda: bool = False

    @property
    def has_gpu(self): return bool(self.gpus)
    @property
    def primary_gpu(self): return self.gpus[0] if self.gpus else None


@dataclass
class AccelConfig:
    backend: str
    n_workers: int
    grid_width: int
    grid_height: int
    description: str
    recommended_steps: int = 1500
    recommended_seeds: int = 20

    def summary(self):
        return "\n".join([
            "=" * 56, "  ACCELERATION CONFIG", "=" * 56,
            f"  Backend      : {self.backend}",
            f"  Workers      : {self.n_workers}",
            f"  Grid         : {self.grid_width}x{self.grid_height}",
            f"  Steps        : {self.recommended_steps}",
            f"  Seeds        : {self.recommended_seeds}",
            f"  Description  : {self.description}",
            "=" * 56,
        ])


# ── Detection ────────────────────────────────────────────────────────────────

def _detect_gpus():
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            for line in r.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    yield GPUInfo(name=parts[0],
                                  memory_mb=int(parts[1]) if parts[1].isdigit() else 0,
                                  compute_cap=parts[2] if len(parts) > 2 else "")
    except Exception:
        pass


def _check_jax():
    try:
        import jax
        try:
            return True, len(jax.devices("gpu")) > 0
        except RuntimeError:
            return True, False
    except ImportError:
        return False, False


def _check_numba():
    try:
        import numba  # noqa
        try:
            from numba import cuda
            return True, cuda.is_available()
        except Exception:
            return True, False
    except ImportError:
        return False, False


def _ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except ImportError:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        return int(line.split()[1]) / 1e6
        except Exception:
            pass
    return 0.0


def detect_hardware() -> HardwareInfo:
    has_jax, has_jax_gpu = _check_jax()
    has_numba, has_numba_cuda = _check_numba()
    try:
        import google.colab  # noqa
        is_colab = True
    except ImportError:
        is_colab = False
    return HardwareInfo(
        cpu_cores=multiprocessing.cpu_count(),
        ram_gb=_ram_gb(),
        gpus=list(_detect_gpus()),
        is_colab=is_colab,
        has_jax=has_jax, has_jax_gpu=has_jax_gpu,
        has_numba=has_numba, has_numba_cuda=has_numba_cuda,
    )


# ── Decision ─────────────────────────────────────────────────────────────────

def get_accel_config(hw: HardwareInfo) -> AccelConfig:
    cores = hw.cpu_cores
    gpu   = hw.primary_gpu

    if gpu:
        tier = gpu.tier
        tiers = {
            "a100": ("jax_gpu",   14, 1500, 20),
            "l4":   ("jax_gpu",   10, 1500, 20),
            "t4":   ("numba_gpu",  6, 1500, 20),
            "v100": ("jax_gpu",    8, 1500, 20),
        }
        backend_pref, w, steps, seeds = tiers.get(tier, ("numba_gpu", 6, 1500, 20))
        # Fall back if preferred library absent
        if "jax" in backend_pref and not hw.has_jax_gpu:
            backend_pref = "numba_gpu" if hw.has_numba_cuda else "numpy"
        if "numba" in backend_pref and not hw.has_numba_cuda:
            backend_pref = "numpy"
        return AccelConfig(backend=backend_pref, n_workers=min(cores, w),
                           grid_width=80, grid_height=80,
                           recommended_steps=steps, recommended_seeds=seeds,
                           description=f"{tier.upper()} {gpu.memory_gb:.0f}GB — {backend_pref}, {min(cores,w)} workers")

    if hw.has_numba and cores >= 8:
        return AccelConfig("numba_cpu", max(1, cores-2), 80, 80,
                           f"CPU {cores}c — Numba parallel, {cores-2} workers", 1000, 20)
    if hw.has_numba:
        return AccelConfig("numba_cpu", max(1, cores-1), 80, 80,
                           f"CPU {cores}c — Numba JIT, {cores-1} workers", 500, 10)
    return AccelConfig("numpy", max(1, cores), 80, 80,
                       f"CPU {cores}c — vectorised NumPy, {cores} workers", 300, 10)


# ── Regen function factory ────────────────────────────────────────────────────
# Signature: regen_fn(food, raw_mat, land, water, pollution) -> None (all in-place)
# Pollution diffusion is applied in environment.step() via NumPy rolls.
# The kernel here only applies per-cell natural decay.

def build_regen_fn(backend: str):
    """
    Return the fastest regen function for the given backend.
    agriculture_rate_mult is applied externally before calling this.
    """

    # ── JAX GPU ──────────────────────────────────────────────────────────────
    if backend == "jax_gpu" and _try_import("jax"):
        import jax, jax.numpy as jnp
        import numpy as np

        @jax.jit
        def _jax(f, r, l, w, p):
            f = jnp.clip(f + FOOD_RATE  * f * (1 - f/FOOD_CAP)  + NUDGE, 0, FOOD_CAP)
            r = jnp.clip(r + RAW_RATE   * r * (1 - r/RAW_CAP)   + NUDGE, 0, RAW_CAP)
            l = jnp.clip(l + LAND_RATE  * l * (1 - l/LAND_CAP)  + NUDGE, 0, LAND_CAP)
            w = jnp.clip(w + WATER_RATE * w * (1 - w/WATER_CAP) + NUDGE, 0, WATER_CAP)
            p = jnp.clip(p * (1 - POLLUTION_DECAY), 0, POLLUTION_CAP)
            return f, r, l, w, p

        def regen_jax(food, raw_mat, land, water, pollution):
            jf, jr, jl, jw, jp = _jax(
                jnp.asarray(food), jnp.asarray(raw_mat),
                jnp.asarray(land), jnp.asarray(water), jnp.asarray(pollution))
            food[:]      = np.asarray(jf)
            raw_mat[:]   = np.asarray(jr)
            land[:]      = np.asarray(jl)
            water[:]     = np.asarray(jw)
            pollution[:] = np.asarray(jp)
        return regen_jax

    # ── Numba CUDA ───────────────────────────────────────────────────────────
    if backend == "numba_gpu" and _try_import("numba"):
        from numba import cuda
        if cuda.is_available():
            try:
                @cuda.jit
                def _cuda_kernel(food, raw_mat, land, water, pollution):
                    x, y = cuda.grid(2)
                    W, H = food.shape
                    if x < W and y < H:
                        f = food[x,y];      food[x,y]      = min(FOOD_CAP,      f + FOOD_RATE*f*(1-f/FOOD_CAP)            + NUDGE)
                        r = raw_mat[x,y];   raw_mat[x,y]   = min(RAW_CAP,       r + RAW_RATE*r*(1-r/RAW_CAP)              + NUDGE)
                        l = land[x,y];      land[x,y]      = min(LAND_CAP,      l + LAND_RATE*l*(1-l/LAND_CAP)            + NUDGE)
                        w = water[x,y];     water[x,y]     = min(WATER_CAP,     w + WATER_RATE*w*(1-w/WATER_CAP)          + NUDGE)
                        p = pollution[x,y]; pollution[x,y] = min(POLLUTION_CAP, max(0.0, p * (1 - POLLUTION_DECAY)))

                def regen_cuda(food, raw_mat, land, water, pollution):
                    W, H = food.shape
                    tpb = (16, 16)
                    bpg = ((W+15)//16, (H+15)//16)
                    df = cuda.to_device(food);      dr = cuda.to_device(raw_mat)
                    dl = cuda.to_device(land);      dw = cuda.to_device(water)
                    dp = cuda.to_device(pollution)
                    _cuda_kernel[bpg, tpb](df, dr, dl, dw, dp)
                    df.copy_to_host(food);      dr.copy_to_host(raw_mat)
                    dl.copy_to_host(land);      dw.copy_to_host(water)
                    dp.copy_to_host(pollution)
                return regen_cuda
            except Exception:
                pass

    # ── Numba CPU ────────────────────────────────────────────────────────────
    if backend in ("numba_cpu", "numba_gpu") and _try_import("numba"):
        try:
            from numba import njit, prange

            @njit(parallel=True, cache=True)
            def _numba(food, raw_mat, land, water, pollution):
                W, H = food.shape
                for x in prange(W):
                    for y in range(H):
                        f = food[x,y];      food[x,y]      = min(FOOD_CAP,      f + FOOD_RATE*f*(1-f/FOOD_CAP)            + NUDGE)
                        r = raw_mat[x,y];   raw_mat[x,y]   = min(RAW_CAP,       r + RAW_RATE*r*(1-r/RAW_CAP)              + NUDGE)
                        l = land[x,y];      land[x,y]      = min(LAND_CAP,      l + LAND_RATE*l*(1-l/LAND_CAP)            + NUDGE)
                        w = water[x,y];     water[x,y]     = min(WATER_CAP,     w + WATER_RATE*w*(1-w/WATER_CAP)          + NUDGE)
                        p = pollution[x,y]; pollution[x,y] = min(POLLUTION_CAP, max(0.0, p * (1 - POLLUTION_DECAY)))

            def regen_numba(food, raw_mat, land, water, pollution):
                _numba(food, raw_mat, land, water, pollution)
            return regen_numba
        except Exception:
            pass

    # ── NumPy fallback ───────────────────────────────────────────────────────
    import numpy as np

    def regen_numpy(food, raw_mat, land, water, pollution):
        np.clip(food    + FOOD_RATE  * food    * (1 - food    / FOOD_CAP)  + NUDGE, 0, FOOD_CAP,      out=food)
        np.clip(raw_mat + RAW_RATE   * raw_mat * (1 - raw_mat / RAW_CAP)   + NUDGE, 0, RAW_CAP,       out=raw_mat)
        np.clip(land    + LAND_RATE  * land    * (1 - land    / LAND_CAP)  + NUDGE, 0, LAND_CAP,      out=land)
        np.clip(water   + WATER_RATE * water   * (1 - water   / WATER_CAP) + NUDGE, 0, WATER_CAP,     out=water)
        np.clip(pollution * (1 - POLLUTION_DECAY),                                  0, POLLUTION_CAP,  out=pollution)
    return regen_numpy


def _try_import(name):
    try: __import__(name); return True
    except ImportError: return False


# ── Convenience ──────────────────────────────────────────────────────────────

_cached_config: Optional[AccelConfig] = None
_cached_hw: Optional[HardwareInfo] = None


def auto_configure(verbose=True) -> AccelConfig:
    global _cached_config, _cached_hw
    if _cached_config is not None:
        return _cached_config
    hw = detect_hardware()
    cfg = get_accel_config(hw)
    _cached_config = cfg
    _cached_hw = hw
    if verbose:
        print(_hardware_report(hw, cfg))
    return cfg


def _hardware_report(hw, cfg):
    lines = ["", "="*56, "  HARDWARE REPORT", "="*56,
             f"  Platform     : {'Google Colab' if hw.is_colab else 'Local'}",
             f"  CPU cores    : {hw.cpu_cores}",
             f"  RAM          : {hw.ram_gb:.1f} GB"]
    for i, g in enumerate(hw.gpus):
        lines.append(f"  GPU {i}         : {g.name} ({g.memory_gb:.0f} GB)")
    if not hw.gpus:
        lines.append("  GPU          : none")
    lines += [
        f"  JAX          : {'yes (GPU)' if hw.has_jax_gpu else 'yes' if hw.has_jax else 'no'}",
        f"  Numba        : {'yes (CUDA)' if hw.has_numba_cuda else 'yes' if hw.has_numba else 'no'}",
        "-"*56, cfg.summary(),
    ]
    return "\n".join(lines)
