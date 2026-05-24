"""Synthetic LEO-like geometry for controlled scenario generation.

The generator intentionally avoids TLE dependencies while still producing
physically interpretable slant ranges and elevation angles.  Satellite
sub-points are arranged into slow pass segments over the service area so that
P2 benchmarks can be feasible without forcing below-threshold links visible.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from leo_alloc.utils.logging import get_logger

logger = get_logger(__name__)

FloatArray = NDArray[np.float64]

_R_EARTH_KM: float = 6371.0
_GM_KM3_S2: float = 3.986004418e5  # km³ s⁻²
_DEG2RAD: float = np.pi / 180.0
_RAD2DEG: float = 180.0 / np.pi


def orbital_period_s(altitude_km: float) -> float:
    """Keplerian period in seconds for a circular LEO orbit."""
    r_km = _R_EARTH_KM + altitude_km
    return float(2.0 * np.pi * np.sqrt(r_km**3 / _GM_KM3_S2))


def generate_cell_positions(
    cell_count: int,
    rng: np.random.Generator,
    lat_range_deg: tuple[float, float] = (45.0, 50.0),
    lon_range_deg: tuple[float, float] = (10.0, 18.0),
) -> tuple[FloatArray, FloatArray]:
    """Return cell latitudes and longitudes sampled in a compact service area.

    Parameters
    ----------
    cell_count : int
        Number of cells.
    rng : Generator
        NumPy random generator.
    lat_range_deg, lon_range_deg : tuple[float, float]
        Latitude / longitude bounds of the service area in degrees.

    Returns
    -------
    lats, lons : ndarray of shape (C,)
        Cell centre coordinates in degrees.
    """
    lats = rng.uniform(lat_range_deg[0], lat_range_deg[1], size=cell_count).astype(np.float64)
    lons = rng.uniform(lon_range_deg[0], lon_range_deg[1], size=cell_count).astype(np.float64)
    return lats, lons


def _satellite_groundtrack(
    omega_rad_s: float,
    phases_rad: FloatArray,  # [S]
    raans_rad: FloatArray,  # [S]
    incl_rad: float,
    time_points_s: FloatArray,  # [K]
) -> tuple[FloatArray, FloatArray]:
    """Sub-satellite latitude and longitude for all satellites and time points.

    Returns
    -------
    sat_lats_deg, sat_lons_deg : ndarray of shape (S, K)
    """
    args = omega_rad_s * time_points_s[None, :] + phases_rad[:, None]  # [S, K]
    sat_lats_rad = np.arcsin(np.clip(np.sin(incl_rad) * np.sin(args), -1.0, 1.0))
    lon_offsets = np.arctan2(np.cos(incl_rad) * np.sin(args), np.cos(args))  # [S, K]
    sat_lons_rad = raans_rad[:, None] + lon_offsets
    sat_lons_deg = (_RAD2DEG * sat_lons_rad + 180.0) % 360.0 - 180.0
    return sat_lats_rad * _RAD2DEG, sat_lons_deg


def _geometry_for_satellite(
    sat_lats_deg: FloatArray,  # [K]
    sat_lons_deg: FloatArray,  # [K]
    altitude_km: float,
    cell_lats_deg: FloatArray,  # [C]
    cell_lons_deg: FloatArray,  # [C]
) -> tuple[FloatArray, FloatArray]:
    """Slant range and elevation for one satellite across all cells and slots.

    Returns
    -------
    distances_m : ndarray of shape (C, K)
    elevations_deg : ndarray of shape (C, K)
    """
    sat_lat = sat_lats_deg[None, :] * _DEG2RAD  # [1, K]
    sat_lon = sat_lons_deg[None, :] * _DEG2RAD  # [1, K]
    cell_lat = cell_lats_deg[:, None] * _DEG2RAD  # [C, 1]
    cell_lon = cell_lons_deg[:, None] * _DEG2RAD  # [C, 1]

    cos_d = np.clip(
        np.sin(cell_lat) * np.sin(sat_lat)
        + np.cos(cell_lat) * np.cos(sat_lat) * np.cos(sat_lon - cell_lon),
        -1.0,
        1.0,
    )  # [C, K]

    r_sat_km = _R_EARTH_KM + altitude_km
    slant_km = np.sqrt(
        r_sat_km**2 + _R_EARTH_KM**2 - 2.0 * r_sat_km * _R_EARTH_KM * cos_d
    )  # [C, K]
    sin_el = np.clip((r_sat_km * cos_d - _R_EARTH_KM) / np.maximum(slant_km, 1e-3), -1.0, 1.0)
    return slant_km * 1e3, np.arcsin(sin_el) * _RAD2DEG  # metres, degrees


def generate_synthetic_geometry(
    satellite_count: int,
    cell_count: int,
    slow_slot_count: int,
    cell_lats_deg: FloatArray,  # [C]
    cell_lons_deg: FloatArray,  # [C]
    rng: np.random.Generator,
    altitude_km: float = 550.0,
    inclination_deg: float = 53.0,
    sim_duration_s: float = 3600.0,
    handover_budget: int | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Compute slant ranges and elevation angles for satellites over slow slots.

    One primary satellite is kept near the service area during each pass segment,
    while the rest follow offset tracks.  The number of pass segments is bounded
    by the handover budget when provided, so the raw visibility pattern is
    usually feasible without artificial coverage repair.

    Parameters
    ----------
    satellite_count, cell_count, slow_slot_count : int
        Number of satellites, cells, slow slots.
    cell_lats_deg, cell_lons_deg : ndarray of shape (C,)
        Cell centre coordinates in degrees.
    rng : Generator
        NumPy random generator.
    altitude_km : float
        Orbital altitude in km.
    inclination_deg : float
        Orbital inclination in degrees.
    sim_duration_s : float
        Total simulation window in seconds.
    handover_budget : int, optional
        Per-cell handover budget used to limit the number of primary pass
        changes in the generated visibility pattern.

    Returns
    -------
    distances_m : ndarray of shape (S, C, K)
        Slant range in metres.
    elevations_deg : ndarray of shape (S, C, K)
        Elevation angle in degrees (negative means below horizon).
    """
    orbital_period = orbital_period_s(altitude_km)
    omega = 2.0 * np.pi / orbital_period
    incl_rad = inclination_deg * _DEG2RAD
    del incl_rad, sim_duration_s  # retained in the public signature for experiment metadata parity

    center_lat = float(np.mean(cell_lats_deg))
    center_lon = float(np.mean(cell_lons_deg))
    span_lat = max(float(np.ptp(cell_lats_deg)), 1.0)
    span_lon = max(float(np.ptp(cell_lons_deg)), 1.0)
    pass_changes = handover_budget if handover_budget is not None else max(satellite_count - 1, 0)
    segment_count = max(1, min(satellite_count, slow_slot_count, pass_changes + 1))
    segment_len = int(np.ceil(slow_slot_count / segment_count))
    slot_index = np.arange(slow_slot_count, dtype=np.float64)

    sat_lats_all = np.zeros((satellite_count, slow_slot_count), dtype=np.float64)
    sat_lons_all = np.zeros((satellite_count, slow_slot_count), dtype=np.float64)
    base_phase = rng.uniform(0.0, 2.0 * np.pi, size=satellite_count)
    pass_phase = omega * slot_index * 120.0

    for sat in range(satellite_count):
        sat_lats_all[sat] = center_lat + 2.0 * span_lat * np.sin(pass_phase + base_phase[sat])
        sat_lons_all[sat] = center_lon + 2.0 * span_lon * np.cos(pass_phase + base_phase[sat])

    for segment in range(segment_count):
        primary_sat = segment % satellite_count
        start = segment * segment_len
        end = min((segment + 1) * segment_len, slow_slot_count)
        if start >= end:
            continue
        local = np.linspace(-1.0, 1.0, end - start)
        sat_lats_all[primary_sat, start:end] = center_lat + 0.15 * span_lat * local
        sat_lons_all[primary_sat, start:end] = center_lon + 0.15 * span_lon * local[::-1]

    sat_lons_all = (sat_lons_all + 180.0) % 360.0 - 180.0

    distances_m = np.zeros((satellite_count, cell_count, slow_slot_count), dtype=np.float64)
    elevations_deg = np.zeros((satellite_count, cell_count, slow_slot_count), dtype=np.float64)
    for sat in range(satellite_count):
        d, el = _geometry_for_satellite(
            sat_lats_all[sat],
            sat_lons_all[sat],
            altitude_km,
            cell_lats_deg,
            cell_lons_deg,
        )
        distances_m[sat] = d
        elevations_deg[sat] = el

    logger.info(
        "Synthetic geometry: S=%d C=%d K=%d, elevation range [%.1f°, %.1f°]",
        satellite_count,
        cell_count,
        slow_slot_count,
        float(np.min(elevations_deg)),
        float(np.max(elevations_deg)),
    )
    return distances_m, elevations_deg
