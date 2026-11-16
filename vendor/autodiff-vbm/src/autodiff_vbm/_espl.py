"""Extended-source point-lens magnification, reproducing VBMicrolensing exactly.

VBMicrolensing evaluates a uniform-brightness finite source over a point lens
by a precomputed table lookup (``ESPL.tbl``), not a contour integral -- that is
what makes ``ESPLMag2`` fast.  This module ports that path
(``VBMicrolensing::ESPLMag`` and ``ESPLMag2``, with ``ESPLMagDark`` collapsing
to a single ``ESPLMag`` call for a uniform source) to pure JAX so the result is
differentiable in ``u`` and ``rho`` while keeping the table-lookup speed.

The table is read from the installed ``VBMicrolensing`` package, so the values
match the C++ engine bit for bit without redistributing it.

``ESPL.tbl`` holds four C-order ``(151, 101)`` ``float64`` arrays back to back:
``ESPLin``, ``ESPLout``, ``ESPLinastro``, ``ESPLoutastro``.  Only the first two
(the magnification tables) are needed for a uniform source; the ``*astro``
tables drive centroid astrometry, which is out of scope here.
"""

from __future__ import annotations

import functools
import pathlib

import jax.numpy as jnp
import numpy as np

# VBMicrolensingLibrary.h
_RSIZE_ESPL = 151
_ZSIZE_ESPL = 101

# VBMicrolensing::ESPLMag: fr = -10.857362047581296 * log(0.01 * RSv).
_ESPL_FR_SCALE = 10.857362047581296


def _espl_table_path() -> pathlib.Path:
    import VBMicrolensing

    path = pathlib.Path(VBMicrolensing.__file__).parent / "data" / "ESPL.tbl"
    if not path.is_file():
        raise RuntimeError(f"VBMicrolensing ESPL table not found at {path}")
    return path


@functools.lru_cache(maxsize=1)
def _load_espl_table():
    """Port of ``VBMicrolensing::LoadESPLTable`` (magnification tables only).

    Returns ``(espl_in, espl_out)``, each ``(151, 101)`` ``float64``.
    """
    raw = np.fromfile(_espl_table_path(), dtype="<f8")
    block = _RSIZE_ESPL * _ZSIZE_ESPL
    if raw.size < 2 * block:
        raise RuntimeError("VBMicrolensing ESPL table is truncated")
    espl_in = raw[:block].reshape(_RSIZE_ESPL, _ZSIZE_ESPL).copy()
    espl_out = raw[block:2 * block].reshape(_RSIZE_ESPL, _ZSIZE_ESPL).copy()
    return espl_in, espl_out


def _bilinear(table, ir, fr, iz, zf):
    """Bilinear cell interpolation, matching ``ESPLMag`` term for term."""
    cr = 1.0 - fr
    cz = 1.0 - zf
    return (
        table[ir, iz] * cr * cz
        + table[ir + 1, iz] * fr * cz
        + table[ir, iz + 1] * cr * zf
        + table[ir + 1, iz + 1] * fr * zf
    )


def _espl_mag_table(u, rho):
    """Port of ``VBMicrolensing::ESPLMag`` (uniform source, no astrometry)."""
    espl_in, espl_out = _load_espl_table()
    t_in = jnp.asarray(espl_in)
    t_out = jnp.asarray(espl_out)

    fr = -_ESPL_FR_SCALE * jnp.log(0.01 * rho)
    fr = jnp.clip(fr, 0.0, _RSIZE_ESPL - 1.000001)
    ir = jnp.clip(jnp.floor(fr).astype(jnp.int32), 0, _RSIZE_ESPL - 2)
    fr = fr - ir

    z = u / rho
    inside = z < 1.0
    z_safe = jnp.where(inside, 1.0, z)

    zz_in = z * (_ZSIZE_ESPL - 1)
    zz_out = (0.99999999999999 / z_safe) * (_ZSIZE_ESPL - 1)
    zz = jnp.where(inside, zz_in, zz_out)
    iz = jnp.clip(jnp.floor(zz).astype(jnp.int32), 0, _ZSIZE_ESPL - 2)
    zf = zz - iz

    corr_in = _bilinear(t_in, ir, fr, iz, zf)
    corr_out = _bilinear(t_out, ir, fr, iz, zf)

    u2 = u * u
    pref_in = jnp.sqrt(1.0 + 4.0 / (rho * rho))
    pref_out = (u2 + 2.0) / jnp.sqrt(u2 * (u2 + 4.0))
    return jnp.where(inside, pref_in * corr_in, pref_out * corr_out)


def espl_magnification(u, rho, *, tol=1e-2, table_only=False):
    """Port of ``VBMicrolensing::ESPLMag2`` for a uniform-brightness source.

    ``ESPLMag2`` switches between a closed-form quadrupole expansion (when the
    quadrupole term ``quad`` is below ``tol/10``) and the table lookup.  The
    switch is a small primal discontinuity; ``table_only=True`` forces the table
    branch everywhere, which removes the parameter-dependent branch for an HMC
    target at the cost of the closed-form branch's accuracy for very small
    ``rho`` (where the table saturates).

    ``u`` and ``rho`` broadcast against each other.
    """
    u = jnp.asarray(u, dtype=jnp.float64)
    rho = jnp.asarray(rho, dtype=jnp.float64)
    u, rho = jnp.broadcast_arrays(u, rho)

    u2 = u * u
    u2_4 = u2 + 4.0
    s_u2_4 = jnp.sqrt(u2_4)
    rho2 = rho * rho
    quad = 4.0 * (u2 + 1.0) * rho2 / (u2 * u * u2_4 * u2_4 * s_u2_4)

    table = _espl_mag_table(u, rho)
    if table_only:
        return table

    analytic = (u2 + 2.0) / (u * s_u2_4) + quad
    return jnp.where(quad * 10.0 < tol, analytic, table)
