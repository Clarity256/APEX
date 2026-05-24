"""Path-loss and log-normal shadow-fading channel gain generation.

All gains are returned in linear scale (not dB). Internal dB conversions are
local to this module; they must not leak into solver or RL code.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from leo_alloc.utils.logging import get_logger

logger = get_logger(__name__)

FloatArray = NDArray[np.float64]

# Free-space path-loss constant: FSPL_dB = 20·log10(d) + 20·log10(f) + C_FSPL
# Derived from: FSPL = (4πdf/c)², c = 3e8 m/s → C = −20·log10(c/(4π)) ≈ −147.55 dB
_C_FSPL: float = -147.55


def _dbi_to_linear(gain_dbi: float) -> float:
    return float(10.0 ** (gain_dbi / 10.0))


def _fspl_linear(distance_m: FloatArray, freq_hz: float) -> FloatArray:
    """Free-space path loss as a linear power ratio (>1, i.e. loss factor)."""
    fspl_db = 20.0 * np.log10(distance_m) + 20.0 * np.log10(freq_hz) + _C_FSPL
    return np.asarray(np.power(10.0, fspl_db / 10.0), dtype=np.float64)


def generate_channel_gains(
    distances_m: FloatArray,
    freq_hz: float,
    g_tx_dbi: float,
    g_rx_dbi: float,
    rng: np.random.Generator,
    atmospheric_loss_db: float = 0.5,
    pointing_loss_db: float = 3.0,
    shadow_std_db: float = 2.0,
) -> FloatArray:
    """Generate the channel gain matrix g[S, C, K] including shadowing.

    The channel model follows doc/05_experiment_design.md:
        gain = G_tx · G_rx · 10^(−(FSPL + shadow + L_atm + L_point) / 10)

    Parameters
    ----------
    distances_m : ndarray of shape (S, C, K)
        Slant range from each satellite to each cell at each slow slot, in metres.
    freq_hz : float
        Carrier frequency in Hz.
    g_tx_dbi, g_rx_dbi : float
        Transmit and receive antenna gains in dBi.
    rng : Generator
        NumPy random generator (no global state).
    atmospheric_loss_db, pointing_loss_db : float
        Atmospheric and pointing losses in dB.
    shadow_std_db : float
        Standard deviation of log-normal shadow fading in dB (0 → no fading).

    Returns
    -------
    ndarray of shape (S, C, K)
        Linear channel gains.
    """
    if np.any(distances_m <= 0.0):
        raise ValueError("distances_m must be strictly positive")

    g_tx = _dbi_to_linear(g_tx_dbi)
    g_rx = _dbi_to_linear(g_rx_dbi)
    fixed_loss_linear = 10.0 ** ((atmospheric_loss_db + pointing_loss_db) / 10.0)

    if shadow_std_db > 0.0:
        shadow_db = rng.normal(0.0, shadow_std_db, size=distances_m.shape)
    else:
        shadow_db = np.zeros(distances_m.shape, dtype=np.float64)
    shadow_linear = np.power(10.0, shadow_db / 10.0)

    gains = (g_tx * g_rx) / (_fspl_linear(distances_m, freq_hz) * fixed_loss_linear * shadow_linear)
    gains = np.asarray(gains, dtype=np.float64)

    logger.debug(
        "Channel gains generated: shape=%s min=%.2e max=%.2e",
        gains.shape,
        float(np.min(gains)),
        float(np.max(gains)),
    )
    return np.asarray(gains, dtype=np.float64)
