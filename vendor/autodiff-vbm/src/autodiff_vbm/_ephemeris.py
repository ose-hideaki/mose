"""Annual-parallax Earth ephemeris, reproducing VBMicrolensing exactly.

VBMicrolensing's default annual-parallax path (``parallaxephemeris = True``,
``t_in_HJD = True``, ``parallaxsystem = 1``, ``t0_par_fixed = 0``) interpolates
the Sun ephemeris table shipped with the wheel and projects Earth's
heliocentric displacement onto the geocentric-equatorial South/West directions
of the target.  This module ports that computation (``VBMicrolensing::
LoadSunTable`` and ``VBMicrolensing::ComputeParallax``) to pure JAX so the
projected offsets ``Et`` are differentiable with respect to ``t0_par``.

The table itself is read from the installed ``VBMicrolensing`` package, so the
values match the C++ engine bit for bit without redistributing JPL data.
"""

from __future__ import annotations

import functools
import pathlib

import jax
import jax.numpy as jnp
import numpy as np


# VBMicrolensing::VBMicrolensing() -- ecliptic basis expressed in the
# J2000 equatorial frame.
_EQ2000 = np.array([1.0, 0.0, 0.0])
_QUAD2000 = np.array([0.0, 0.9174820003578725, -0.3977772982704228])
_NORTH2000 = np.array([0.0, 0.3977772982704228, 0.9174820003578725])

_AU_C_DAY = 0.005775518331436995  # au / c in days


def _sun_table_path() -> pathlib.Path:
    import VBMicrolensing

    installed_path = (
        pathlib.Path(VBMicrolensing.__file__).parent / "data" / "SunEphemeris.txt"
    )
    if installed_path.is_file():
        return installed_path

    bundled_path = pathlib.Path(__file__).parent / "data" / "SunEphemeris.txt"
    if bundled_path.is_file():
        return bundled_path

    raise RuntimeError(
        "VBMicrolensing Sun ephemeris was not found at either "
        f"{installed_path} or {bundled_path}"
    )


@functools.lru_cache(maxsize=1)
def _load_sun_table():
    """Port of ``VBMicrolensing::LoadSunTable``.

    Returns ``(pos_earth, start_reduced_jd, step)`` where ``pos_earth`` has
    shape ``(N, 3)`` and is Earth's heliocentric position in au in the J2000
    equatorial frame (VBM stores ``-dist * sun_unit_vector``).
    """
    rows = []
    in_block = False
    with _sun_table_path().open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not in_block:
                if stripped.startswith("$$SOE"):
                    in_block = True
                continue
            if stripped.startswith("$$EOE"):
                break
            parts = stripped.split()
            if len(parts) < 4:
                continue
            jd, ra_deg, dec_deg, dist = (float(parts[i]) for i in range(4))
            rows.append((jd, ra_deg, dec_deg, dist))

    if len(rows) < 3:
        raise RuntimeError("VBMicrolensing Sun ephemeris table is too short")

    table = np.asarray(rows, dtype=np.float64)
    jd = table[:, 0]
    ra = np.deg2rad(table[:, 1])
    dec = np.deg2rad(table[:, 2])
    dist = table[:, 3]

    unit = (
        np.cos(ra)[:, None] * np.cos(dec)[:, None] * _EQ2000[None, :]
        + np.sin(ra)[:, None] * np.cos(dec)[:, None] * _QUAD2000[None, :]
        + np.sin(dec)[:, None] * _NORTH2000[None, :]
    )
    pos_earth = -dist[:, None] * unit

    start = jd[0] - 2450000.0
    step = jd[1] - jd[0]
    return pos_earth, float(start), float(step)


def parse_coordinate_string(coordinate_string: str) -> tuple[float, float]:
    """Port of ``VBMicrolensing::SetObjectCoordinates(char* CoordinateString)``.

    Accepts ``"hr:mn:sc deg:pr:sc"`` and returns ``(ra_rad, dec_rad)``.
    """
    ra_part, dec_part = coordinate_string.split()
    hr, mn, sc = (float(x) for x in ra_part.split(":"))
    deg, pr, ssc = (float(x) for x in dec_part.split(":"))
    ra = (hr + mn / 60.0 + sc / 3600.0) * np.pi / 12.0
    dec = (abs(deg) + pr / 60.0 + ssc / 3600.0) * np.pi / 180.0
    if deg < 0:
        dec = -dec
    return float(ra), float(dec)


def projection_basis(ra_rad: float, dec_rad: float):
    """Port of the ``Obj``/``rad``/``tang`` construction in SetObjectCoordinates.

    Returns ``(obj, south, west)`` unit vectors in the J2000 equatorial frame,
    where ``south`` and ``west`` span the plane orthogonal to the line of
    sight (``parallaxsystem = 1``).
    """
    obj = (
        np.cos(ra_rad) * np.cos(dec_rad) * _EQ2000
        + np.sin(ra_rad) * np.cos(dec_rad) * _QUAD2000
        + np.sin(dec_rad) * _NORTH2000
    )
    sp = float(_NORTH2000 @ obj)
    south = -_NORTH2000 + sp * obj
    south = south / np.linalg.norm(south)
    west = np.cross(south, obj)
    return obj, south, west


def _interp_pos(pos_earth, start, step, x):
    """Linear interpolation of ``pos_earth`` at reduced-JD query ``x``."""
    n = pos_earth.shape[0]
    u = (x - start) / step
    i0 = jnp.clip(jnp.floor(u).astype(jnp.int32), 0, n - 2)
    w = u - i0.astype(u.dtype)
    p0 = pos_earth[i0]
    p1 = pos_earth[i0 + 1]
    return p0 + (p1 - p0) * w[..., None]


def _emission_time(pos_earth, start, step, obj, query, iters):
    """Converged light emission time: ``jd`` such that ``jd = query - lt(jd)``.

    ``t_in_HJD = True`` means the observed time is heliocentric, so Earth's
    position must be interpolated at the retarded time.  VBMicrolensing runs
    this fixed point to ``1e-8`` day; a handful of iterations reaches that.
    """
    def body(_, jd):
        pos = _interp_pos(pos_earth, start, step, jd)
        lt = (pos @ obj) * _AU_C_DAY
        return query - lt

    return jax.lax.fori_loop(0, iters, body, query)


def _velocity_blend(pos_earth, start, step, jd):
    """Port of the 3-point ``vEar`` blend, evaluated at emission time ``jd``.

    VBMicrolensing computes the Earth velocity for the ``t0_par`` extrapolation
    term from the same retarded ``ic``/``ty`` the position interpolation
    converged to, switching between a backward- and forward-leaning stencil at
    the half-step.
    """
    n = pos_earth.shape[0]
    u = (jd - start) / step
    ic = jnp.clip(jnp.floor(u).astype(jnp.int32), 1, n - 3)
    ty = u - ic.astype(u.dtype)

    p_m1 = pos_earth[ic - 1]
    p_0 = pos_earth[ic]
    p_1 = pos_earth[ic + 1]
    p_2 = pos_earth[ic + 2]

    hi = ((p_2 - p_1) * (ty - 0.5) + (p_1 - p_0) * (1.5 - ty)) / step
    lo = ((p_0 - p_m1) * (0.5 - ty) + (p_1 - p_0) * (ty + 0.5)) / step
    return jnp.where(ty > 0.5, hi, lo)


def earth_projected_offsets(times, t0_par, ra_rad, dec_rad, *, light_time_iters=6):
    """Port of ``VBMicrolensing::ComputeParallax`` (default HJD table path).

    Returns ``Et`` with shape ``(len(times), 2)``: Earth's displacement,
    projected on ``(South, West)``, relative to the linear extrapolation from
    ``t0_par``.  The light curve then uses
    ``tn += pai1 * Et[:, 0] + pai2 * Et[:, 1]`` and
    ``u += pai1 * Et[:, 1] - pai2 * Et[:, 0]``.
    """
    pos_np, start, step = _load_sun_table()
    pos_earth = jnp.asarray(pos_np)
    obj, south, west = projection_basis(ra_rad, dec_rad)
    obj = jnp.asarray(obj)
    south = jnp.asarray(south)
    west = jnp.asarray(west)

    times = jnp.asarray(times, dtype=jnp.float64)
    t0_par = jnp.asarray(t0_par, dtype=jnp.float64)

    jd0 = _emission_time(pos_earth, start, step, obj, t0_par, light_time_iters)
    ear_ref = _interp_pos(pos_earth, start, step, jd0)
    vear_ref = _velocity_blend(pos_earth, start, step, jd0)

    et0_south = ear_ref @ south
    et0_west = ear_ref @ west
    vt0_south = vear_ref @ south
    vt0_west = vear_ref @ west

    jd_t = jax.vmap(
        lambda q: _emission_time(pos_earth, start, step, obj, q, light_time_iters)
    )(times)
    ear_t = jax.vmap(lambda j: _interp_pos(pos_earth, start, step, j))(jd_t)
    ehel_south = ear_t @ south
    ehel_west = ear_t @ west

    et_south = ehel_south - et0_south - vt0_south * (times - t0_par)
    et_west = ehel_west - et0_west - vt0_west * (times - t0_par)
    return jnp.stack([et_south, et_west], axis=-1)
