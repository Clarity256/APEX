"""ScenarioInstance — canonical immutable container for one simulation window."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ScenarioInstance:
    """Immutable scenario object passed to all solvers and the RL environment.

    All arrays use the conventions from doc/00_research_context.md.
    Do NOT modify this class without explicit user approval (see CLAUDE.md).
    """

    # Dimensions
    S: int  # number of satellites
    C: int  # number of cells
    K: int  # number of slow slots
    M: int  # number of fast slots per slow slot

    # Time-varying tensors
    g: FloatArray  # channel gain, shape [S, C, K], linear (not dB)
    v: FloatArray  # visibility, shape [S, C, K], values in {0.0, 1.0}
    a: FloatArray  # demand_arrival, shape [C, K, M], bits

    # Per-satellite system constants
    N_PRB: FloatArray  # PRB pool per satellite, shape [S]
    P_max: FloatArray  # max transmit power per satellite, shape [S], Watts

    # Per-cell system constants
    H: FloatArray  # handover budget per cell, shape [C]

    # Scalar physical constants
    W_PRB: float  # single-PRB bandwidth, Hz
    N0: float  # noise power spectral density, W/Hz
    T_f: float  # fast-slot duration, seconds
    eps: float  # log-utility regularisation term
    lambda_h: float  # soft handover penalty coefficient

    # Metadata
    seed: int  # random seed for reproducibility
    scenario_id: str  # human-readable identifier

    def __post_init__(self) -> None:
        """Validate and freeze array fields after dataclass construction."""
        _require_positive_int(self.S, "S")
        _require_positive_int(self.C, "C")
        _require_positive_int(self.K, "K")
        _require_positive_int(self.M, "M")
        object.__setattr__(self, "g", _array(self.g, "g", (self.S, self.C, self.K), 0.0))
        object.__setattr__(self, "v", _binary_array(self.v, "v", (self.S, self.C, self.K)))
        object.__setattr__(self, "a", _array(self.a, "a", (self.C, self.K, self.M), 0.0))
        object.__setattr__(self, "N_PRB", _array(self.N_PRB, "N_PRB", (self.S,), 0.0, True))
        object.__setattr__(self, "P_max", _array(self.P_max, "P_max", (self.S,), 0.0, True))
        object.__setattr__(self, "H", _array(self.H, "H", (self.C,), 0.0))
        _require_positive_float(self.W_PRB, "W_PRB")
        _require_positive_float(self.N0, "N0")
        _require_positive_float(self.T_f, "T_f")
        _require_positive_float(self.eps, "eps")
        if not np.isfinite(self.lambda_h) or self.lambda_h < 0.0:
            raise ValueError("lambda_h must be a non-negative finite scalar")
        if not self.scenario_id:
            raise ValueError("scenario_id must be non-empty")


def _require_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_positive_float(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite scalar")


def _array(
    value: object,
    name: str,
    shape: tuple[int, ...],
    lower_bound: float,
    strict: bool = False,
) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    if strict:
        invalid = array <= lower_bound
        message = f"{name} must be greater than {lower_bound}"
    else:
        invalid = array < lower_bound
        message = f"{name} must be at least {lower_bound}"
    if np.any(invalid):
        raise ValueError(message)
    array.setflags(write=False)
    return array


def _binary_array(value: object, name: str, shape: tuple[int, ...]) -> FloatArray:
    array = _array(value, name, shape, 0.0)
    if not np.all(np.isclose(array, 0.0) | np.isclose(array, 1.0)):
        raise ValueError(f"{name} must be binary")
    rounded = np.array(np.round(array), dtype=np.float64, copy=True)
    rounded.setflags(write=False)
    return rounded
