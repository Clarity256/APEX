"""Tests for scenario/channel.py — path-loss and shadow-fading channel gains."""

from __future__ import annotations

import numpy as np
import pytest

from leo_alloc.scenario.channel import generate_channel_gains

_FREQ_HZ = 2e9
_G_TX_dBi = 30.0
_G_RX_dBi = 0.0


def _gains(
    distances_m: np.ndarray,
    seed: int = 0,
    shadow_std_db: float = 0.0,
) -> np.ndarray:
    return generate_channel_gains(
        distances_m=distances_m,
        freq_hz=_FREQ_HZ,
        g_tx_dbi=_G_TX_dBi,
        g_rx_dbi=_G_RX_dBi,
        rng=np.random.default_rng(seed),
        shadow_std_db=shadow_std_db,
    )


def test_channel_gains_shape() -> None:
    d = np.full((3, 5, 7), 1e6, dtype=np.float64)
    g = _gains(d)
    assert g.shape == (3, 5, 7)


def test_channel_gains_positive() -> None:
    d = np.full((2, 4, 6), 800e3, dtype=np.float64)
    g = _gains(d)
    assert np.all(g > 0.0)


def test_channel_gains_dtype_float64() -> None:
    d = np.full((2, 2, 2), 1e6, dtype=np.float64)
    g = _gains(d)
    assert g.dtype == np.float64


def test_larger_distance_gives_smaller_gain() -> None:
    d_near = np.full((1, 1, 1), 600e3, dtype=np.float64)
    d_far = np.full((1, 1, 1), 2000e3, dtype=np.float64)
    g_near = _gains(d_near, shadow_std_db=0.0)
    g_far = _gains(d_far, shadow_std_db=0.0)
    assert float(g_near[0, 0, 0]) > float(g_far[0, 0, 0])


def test_channel_gains_reproducible() -> None:
    d = np.random.default_rng(5).uniform(600e3, 2000e3, (2, 3, 4))
    g1 = _gains(d, seed=42, shadow_std_db=2.0)
    g2 = _gains(d, seed=42, shadow_std_db=2.0)
    np.testing.assert_array_equal(g1, g2)


def test_channel_gains_differ_with_different_seeds() -> None:
    d = np.full((2, 3, 4), 1e6, dtype=np.float64)
    g1 = _gains(d, seed=0, shadow_std_db=2.0)
    g2 = _gains(d, seed=1, shadow_std_db=2.0)
    # values are ~1e-13; use array_equal (not allclose whose atol would swamp them)
    assert not np.array_equal(g1, g2)


def test_zero_shadow_std_gives_deterministic_result() -> None:
    d = np.full((2, 3, 4), 1e6, dtype=np.float64)
    g1 = _gains(d, seed=0, shadow_std_db=0.0)
    g2 = _gains(d, seed=99, shadow_std_db=0.0)
    np.testing.assert_allclose(g1, g2, rtol=1e-12)


def test_nonpositive_distance_raises() -> None:
    d = np.array([[[0.0]]], dtype=np.float64)
    with pytest.raises(ValueError):
        _gains(d)
