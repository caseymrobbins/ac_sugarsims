"""
hardware.py
-----------
Hardware detection and acceleration configuration.

Detects:
  - CPU cores and RAM
  - GPU type and VRAM (A100, L4, T4, V100, or generic)
  - Available acceleration libraries (JAX, Numba, CuPy)
  - Whether running in Google Colab

Returns an AccelConfig that the rest of the codebase uses to pick the
fastest resource-grid implementation and worker count.

Decision table
--------------
Hardware           | Backend   | Workers | Grid    | Notes
-------------------|-----------|---------|---------|---------------------------
A100 (40-80 GB)    | jax_gpu   | 14      | 80x80   | JAX JIT on GPU, vmap later
L4  (24 GB)        | jax_gpu   | 10      | 80x80   | JAX JIT on GPU
T4  (16 GB)        | numba_gpu | 6       | 80x80   | Numba CUDA
V100 (16-32 GB)    | jax_gpu   | 8       | 80x80   | JAX JIT on GPU
Generic GPU        | numba_gpu | 6       | 80x80   | Numba CUDA
CPU (8+ cores)     | numba_cpu | n-2     | 80x80   | Numba parallel @njit
CPU (<8 cores)     | numpy     | cores   | 80x80   | Vectorised NumPy (no extra deps)
"""

from __future__ import annotations

import multiprocessing
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GPUInfo:
    name: str
    memory_mb: int
    compute_cap: str = ""

    @property
    def memory_gb(self) -> float:
        return self.memory_mb / 1024

    @property
    def is_a100(self) -> bool:
        return "a100" in self.name.lower()

    @property
    def is_l4(self) -> bool:
        return "l4" in self.name.lower()

    @property
    def is_t4(self) -> bool:
        return "t4" in self.name.lower()

    @property
    def is_v100(self) -> bool:
        return "v100" in self.name.lower()

    @property
    def tier(self) -> str:
        if self.is_a100:
            return "a100"
        if self.is_l4:
            return "l4"
        if self.is_t4:
            return "t4"
        if self.is_v100:
            return "v100"
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
    def has_gpu(self) -> bool:
        return len(self.gpus) > 0

    @property
    def primary_gpu(self) -> Optional[GPUInfo]:
        return self.gpus[0] if self.gpus else None


@dataclass
class AccelConfig:
    backend: str          # "numpy" | "numba_cpu" | "numba_gpu" | "jax_gpu"
    n_workers: int        # parallel episode worker processes
    grid_width: int
    grid_height: int
    description: str
    recommended_steps: int = 1500
    recommended_seeds: int = 20

    def summary(self) -> str:
        lines = [
            "=" * 56,
            "  ACCELERATION CONFIG",
            "=" * 56,
            f"  Backend      : {self.backend}",
            f"  Workers      : {self.n_workers}",
            f"  Grid         : {self.grid_width}x{self.grid_height}",
            f"  Steps        : {self.recommended_steps}",
            f"  Seeds        : {self.recommended_seeds}",
            f"  Description  : {self.description}",
            "=" * 56,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _detect_gpus() -> List[GPUInfo]:
    """Query nvidia-smi for GPU info."""
    gpus = []
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return gpus
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                name = parts[0]
                try:
                    mem_mb = int(parts[1])
                except ValueError:
                    mem_mb = 0
                cap = parts[2] if len(parts) > 2 else ""
                gpus.append(GPUInfo(name=name, memory_mb=mem_mb, compute_cap=cap))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return gpus


def _check_jax() -> tuple[bool, bool]:
    """Return (has_jax, has_jax_gpu)."""
    try:
        import jax
        import jax.numpy as jnp
        # Try a GPU op
        try:
            devices = jax.devices("gpu")
            return True, len(devices) > 0
        except RuntimeError:
            return True, False
    except ImportError:
        return False, False


def _check_numba() -> tuple[bool, bool]:
    """Return (has_numba, has_numba_cuda)."""
    try:
        import numba  # noqa: F401
        has_numba = True
    except ImportError:
        return False, False
    try:
        from numba import cuda
        has_cuda = cuda.is_available()
        return True, has_cuda
    except Exception:
        return True, False


def _check_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _ram_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except ImportError:
        # Fallback: read /proc/meminfo on Linux
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        return kb / 1e6
        except Exception:
            pass
    return 0.0


def detect_hardware() -> HardwareInfo:
    """Detect available hardware and acceleration libraries."""
    cpu_cores = multiprocessing.cpu_count()
    ram_gb = _ram_gb()
    gpus = _detect_gpus()
    is_colab = _check_colab()
    has_jax, has_jax_gpu = _check_jax()
    has_numba, has_numba_cuda = _check_numba()

    return HardwareInfo(
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
        gpus=gpus,
        is_colab=is_colab,
        has_jax=has_jax,
        has_jax_gpu=has_jax_gpu,
        has_numba=has_numba,
        has_numba_cuda=has_numba_cuda,
    )


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

def get_accel_config(hw: HardwareInfo) -> AccelConfig:
    """
    Map detected hardware to the best acceleration strategy.
    """
    cores = hw.cpu_cores
    gpu = hw.primary_gpu

    # -----------------------------------------------------------------------
    # GPU path
    # -----------------------------------------------------------------------
    if gpu is not None:
        tier = gpu.tier

        if tier == "a100":
            backend = "jax_gpu" if hw.has_jax_gpu else ("numba_gpu" if hw.has_numba_cuda else "numpy")
            return AccelConfig(
                backend=backend,
                n_workers=min(cores, 14),
                grid_width=80, grid_height=80,
                recommended_steps=1500, recommended_seeds=20,
                description=f"A100 {gpu.memory_gb:.0f} GB — {backend}, {min(cores,14)} workers",
            )

        if tier == "l4":
            backend = "jax_gpu" if hw.has_jax_gpu else ("numba_gpu" if hw.has_numba_cuda else "numpy")
            return AccelConfig(
                backend=backend,
                n_workers=min(cores, 10),
                grid_width=80, grid_height=80,
                recommended_steps=1500, recommended_seeds=20,
                description=f"L4 {gpu.memory_gb:.0f} GB — {backend}, {min(cores,10)} workers",
            )

        if tier == "t4":
            backend = "numba_gpu" if hw.has_numba_cuda else ("jax_gpu" if hw.has_jax_gpu else "numpy")
            return AccelConfig(
                backend=backend,
                n_workers=min(cores, 6),
                grid_width=80, grid_height=80,
                recommended_steps=1500, recommended_seeds=20,
                description=f"T4 {gpu.memory_gb:.0f} GB — {backend}, {min(cores,6)} workers",
            )

        if tier == "v100":
            backend = "jax_gpu" if hw.has_jax_gpu else ("numba_gpu" if hw.has_numba_cuda else "numpy")
            return AccelConfig(
                backend=backend,
                n_workers=min(cores, 8),
                grid_width=80, grid_height=80,
                recommended_steps=1500, recommended_seeds=20,
                description=f"V100 {gpu.memory_gb:.0f} GB — {backend}, {min(cores,8)} workers",
            )

        # Generic GPU
        backend = "numba_gpu" if hw.has_numba_cuda else ("jax_gpu" if hw.has_jax_gpu else "numpy")
        return AccelConfig(
            backend=backend,
            n_workers=min(cores, 6),
            grid_width=80, grid_height=80,
            recommended_steps=1500, recommended_seeds=20,
            description=f"GPU {gpu.name} — {backend}, {min(cores,6)} workers",
        )

    # -----------------------------------------------------------------------
    # CPU-only path
    # -----------------------------------------------------------------------
    if hw.has_numba and cores >= 8:
        backend = "numba_cpu"
        workers = max(1, cores - 2)
        steps = 1000
        seeds = 20
        desc = f"CPU {cores} cores — Numba JIT parallel, {workers} workers"
    elif hw.has_numba and cores >= 4:
        backend = "numba_cpu"
        workers = max(1, cores - 1)
        steps = 500
        seeds = 10
        desc = f"CPU {cores} cores — Numba JIT, {workers} workers"
    else:
        backend = "numpy"
        workers = max(1, cores)
        steps = 300
        seeds = 10
        desc = f"CPU {cores} cores — vectorised NumPy, {workers} workers"

    return AccelConfig(
        backend=backend,
        n_workers=workers,
        grid_width=80, grid_height=80,
        recommended_steps=steps,
        recommended_seeds=seeds,
        description=desc,
    )


# ---------------------------------------------------------------------------
# Resource regeneration implementations
# ---------------------------------------------------------------------------

def build_regen_fn(backend: str):
    """
    Return the fastest available resource regeneration function
    for the given backend string.

    Signature: regen_fn(food, raw_mat, land) -> None  (in-place)
    All arrays are shape (W, H), dtype float64.
    """

    # Constants baked in for speed
    FOOD_CAP, FOOD_RATE = 30.0, 0.15
    RAW_CAP, RAW_RATE   = 25.0, 0.05
    LAND_CAP, LAND_RATE = 20.0, 0.01
    NUDGE = 0.01

    # --- JAX GPU/CPU -------------------------------------------------------
    if backend in ("jax_gpu",) and _try_import("jax"):
        import jax
        import jax.numpy as jnp

        @jax.jit
        def _jax_regen(food, raw_mat, land):
            food    = jnp.clip(food    + FOOD_RATE * food    * (1.0 - food    / FOOD_CAP) + NUDGE, 0, FOOD_CAP)
            raw_mat = jnp.clip(raw_mat + RAW_RATE  * raw_mat * (1.0 - raw_mat / RAW_CAP)  + NUDGE, 0, RAW_CAP)
            land    = jnp.clip(land    + LAND_RATE * land    * (1.0 - land    / LAND_CAP) + NUDGE, 0, LAND_CAP)
            return food, raw_mat, land

        def regen_jax(food, raw_mat, land):
            # Move to jax arrays, run, copy back
            import numpy as np
            jf, jr, jl = _jax_regen(
                jnp.asarray(food), jnp.asarray(raw_mat), jnp.asarray(land))
            food[:]    = np.asarray(jf)
            raw_mat[:] = np.asarray(jr)
            land[:]    = np.asarray(jl)

        return regen_jax

    # --- Numba GPU (CUDA) --------------------------------------------------
    if backend == "numba_gpu" and _try_import("numba"):
        from numba import cuda
        if cuda.is_available():
            try:
                import numpy as np

                @cuda.jit
                def _cuda_kernel(food, raw_mat, land):
                    x, y = cuda.grid(2)
                    W, H = food.shape
                    if x < W and y < H:
                        f = food[x, y]
                        food[x, y] = min(FOOD_CAP,
                            f + FOOD_RATE * f * (1.0 - f / FOOD_CAP) + NUDGE)
                        r = raw_mat[x, y]
                        raw_mat[x, y] = min(RAW_CAP,
                            r + RAW_RATE * r * (1.0 - r / RAW_CAP) + NUDGE)
                        l = land[x, y]
                        land[x, y] = min(LAND_CAP,
                            l + LAND_RATE * l * (1.0 - l / LAND_CAP) + NUDGE)

                def regen_cuda(food, raw_mat, land):
                    W, H = food.shape
                    tpb = (16, 16)
                    bpg = ((W + tpb[0] - 1) // tpb[0],
                           (H + tpb[1] - 1) // tpb[1])
                    d_food    = cuda.to_device(food)
                    d_raw_mat = cuda.to_device(raw_mat)
                    d_land    = cuda.to_device(land)
                    _cuda_kernel[bpg, tpb](d_food, d_raw_mat, d_land)
                    d_food.copy_to_host(food)
                    d_raw_mat.copy_to_host(raw_mat)
                    d_land.copy_to_host(land)

                return regen_cuda
            except Exception:
                pass  # fall through to numba_cpu

    # --- Numba CPU (parallel JIT) ------------------------------------------
    if backend in ("numba_cpu", "numba_gpu") and _try_import("numba"):
        try:
            from numba import njit, prange

            @njit(parallel=True, cache=True)
            def _numba_regen(food, raw_mat, land):
                W, H = food.shape
                for x in prange(W):
                    for y in range(H):
                        f = food[x, y]
                        food[x, y] = min(FOOD_CAP,
                            f + FOOD_RATE * f * (1.0 - f / FOOD_CAP) + NUDGE)
                        r = raw_mat[x, y]
                        raw_mat[x, y] = min(RAW_CAP,
                            r + RAW_RATE * r * (1.0 - r / RAW_CAP) + NUDGE)
                        l = land[x, y]
                        land[x, y] = min(LAND_CAP,
                            l + LAND_RATE * l * (1.0 - l / LAND_CAP) + NUDGE)

            def regen_numba(food, raw_mat, land):
                _numba_regen(food, raw_mat, land)

            return regen_numba
        except Exception:
            pass

    # --- Vectorised NumPy (fallback, always available) ---------------------
    import numpy as np

    def regen_numpy(food, raw_mat, land):
        np.clip(food + FOOD_RATE * food * (1.0 - food / FOOD_CAP) + NUDGE,
                0, FOOD_CAP, out=food)
        np.clip(raw_mat + RAW_RATE * raw_mat * (1.0 - raw_mat / RAW_CAP) + NUDGE,
                0, RAW_CAP, out=raw_mat)
        np.clip(land + LAND_RATE * land * (1.0 - land / LAND_CAP) + NUDGE,
                0, LAND_CAP, out=land)

    return regen_numpy


def _try_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Convenience: detect + configure in one call
# ---------------------------------------------------------------------------

_cached_config: Optional[AccelConfig] = None
_cached_hw: Optional[HardwareInfo] = None


def auto_configure(verbose: bool = True) -> AccelConfig:
    """
    Detect hardware and return the best AccelConfig.
    Results are cached after the first call.
    """
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


def _hardware_report(hw: HardwareInfo, cfg: AccelConfig) -> str:
    lines = [
        "",
        "=" * 56,
        "  HARDWARE REPORT",
        "=" * 56,
        f"  Platform     : {'Google Colab' if hw.is_colab else 'Local / other'}",
        f"  CPU cores    : {hw.cpu_cores}",
        f"  RAM          : {hw.ram_gb:.1f} GB",
    ]
    if hw.gpus:
        for i, g in enumerate(hw.gpus):
            lines.append(f"  GPU {i}         : {g.name} ({g.memory_gb:.0f} GB)")
    else:
        lines.append("  GPU          : none detected")

    lines += [
        f"  JAX          : {'yes (GPU)' if hw.has_jax_gpu else 'yes (CPU)' if hw.has_jax else 'no'}",
        f"  Numba        : {'yes (CUDA)' if hw.has_numba_cuda else 'yes (CPU)' if hw.has_numba else 'no'}",
        "-" * 56,
        cfg.summary(),
    ]
    return "\n".join(lines)
