"""Tests for scenario/demand.py — Poisson demand arrival generation."""

from __future__ import annotations

import numpy as np
import pytest

from leo_alloc.scenario.demand import generate_demand


@pytest.fixture
def default_kwargs() -> dict:
    return dict(
        mismatch_ratio=0.4,
        demand_base_high_bps=50e6,
        demand_base_low_bps=10e6,
        t_fast_slot_s=0.01,
    )


def test_demand_shape_matches_dimensions(default_kwargs: dict) -> None:
    rng = np.random.default_rng(0)
    a = generate_demand(
        cell_count=5, slow_slot_count=8, fast_slot_count=4, rng=rng, **default_kwargs
    )
    assert a.shape == (5, 8, 4)


def test_demand_nonnegative(default_kwargs: dict) -> None:
    rng = np.random.default_rng(1)
    a = generate_demand(
        cell_count=6, slow_slot_count=10, fast_slot_count=5, rng=rng, **default_kwargs
    )
    assert np.all(a >= 0.0)


def test_demand_dtype_is_float64(default_kwargs: dict) -> None:
    rng = np.random.default_rng(2)
    a = generate_demand(
        cell_count=4, slow_slot_count=6, fast_slot_count=3, rng=rng, **default_kwargs
    )
    assert a.dtype == np.float64


def test_high_load_cells_have_larger_mean(default_kwargs: dict) -> None:
    cell_count, slow_slot_count, fast_slot_count = 10, 20, 10
    rng = np.random.default_rng(42)
    a = generate_demand(
        cell_count=cell_count,
        slow_slot_count=slow_slot_count,
        fast_slot_count=fast_slot_count,
        rng=rng,
        **default_kwargs,
    )
    per_cell_mean = np.mean(a, axis=(1, 2))
    high_count = int(cell_count * default_kwargs["mismatch_ratio"])
    assert high_count > 0 and high_count < cell_count
    assert float(np.mean(np.sort(per_cell_mean)[-high_count:])) > float(
        np.mean(np.sort(per_cell_mean)[: cell_count - high_count])
    )


def test_demand_reproducible_with_same_seed(default_kwargs: dict) -> None:
    a1 = generate_demand(
        cell_count=5,
        slow_slot_count=10,
        fast_slot_count=4,
        rng=np.random.default_rng(7),
        **default_kwargs,
    )
    a2 = generate_demand(
        cell_count=5,
        slow_slot_count=10,
        fast_slot_count=4,
        rng=np.random.default_rng(7),
        **default_kwargs,
    )
    np.testing.assert_array_equal(a1, a2)


def test_demand_differs_with_different_seeds(default_kwargs: dict) -> None:
    a1 = generate_demand(
        cell_count=5,
        slow_slot_count=10,
        fast_slot_count=4,
        rng=np.random.default_rng(0),
        **default_kwargs,
    )
    a2 = generate_demand(
        cell_count=5,
        slow_slot_count=10,
        fast_slot_count=4,
        rng=np.random.default_rng(1),
        **default_kwargs,
    )
    assert not np.array_equal(a1, a2)


def test_zero_mismatch_ratio_all_cells_low_load() -> None:
    rng = np.random.default_rng(3)
    low_bps = 10e6
    a = generate_demand(
        cell_count=6,
        slow_slot_count=20,
        fast_slot_count=10,
        mismatch_ratio=0.0,
        demand_base_high_bps=100e6,
        demand_base_low_bps=low_bps,
        t_fast_slot_s=0.01,
        rng=rng,
    )
    expected_mean_per_slot = low_bps * 0.01
    assert float(np.mean(a)) == pytest.approx(expected_mean_per_slot, rel=0.1)


def test_invalid_mismatch_ratio_raises() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        generate_demand(
            cell_count=4,
            slow_slot_count=5,
            fast_slot_count=3,
            mismatch_ratio=1.5,
            demand_base_high_bps=50e6,
            demand_base_low_bps=10e6,
            t_fast_slot_s=0.01,
            rng=rng,
        )
