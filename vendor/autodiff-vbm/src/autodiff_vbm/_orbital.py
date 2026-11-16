"""Lens orbital-motion trajectories, ported from VBMicrolensing 5.5.

These reproduce ``VBMicrolensing::BinaryAstroLightCurveOrbital`` (circular,
first-order-consistent 3D motion) and
``VBMicrolensing::BinaryAstroLightCurveKepler`` (full Keplerian motion) with the
``astrometry = False`` and ``t_in_HJD = True`` defaults, i.e. no light-travel
term.  Each returns per-epoch source coordinates ``(y1, y2)`` in the binary
centre-of-mass frame plus the projected lens separation ``sep``.

All operations are JAX-traceable; the Kepler eccentric-anomaly solve uses a
fixed Newton iteration count.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _wrap_pi(m):
    return m - 2.0 * jnp.pi * jnp.round(m / (2.0 * jnp.pi))


def circular_trajectory(times, params, et):
    """Port of ``BinaryAstroLightCurveOrbital``.

    ``params`` = ``[log_s, log_q, u0, alpha_0, log_rho, log_tE, t0, pai1, pai2,
    w1, w2, w3]``; ``et`` has shape ``(N, 2)`` = ``(South, West)`` Earth offset.
    Returns ``(y1, y2, sep)``.
    """
    (log_s, _log_q, u0, alpha, _log_rho, log_tE, t0, pai1, pai2, w1, w2, w3) = (
        params[i] for i in range(12)
    )
    s = jnp.exp(log_s)
    tE_inv = jnp.exp(-log_tE)
    salpha = jnp.sin(alpha)
    calpha = jnp.cos(alpha)

    w13sq = w1 * w1 + w3 * w3
    w123 = jnp.sqrt(w13sq + w2 * w2)
    w13 = jnp.sqrt(w13sq)
    degenerate = w13 <= 1e-8

    w13_safe = jnp.where(degenerate, 1.0, w13)
    w123_safe = jnp.where(w123 <= 0.0, 1.0, w123)
    w3c = jnp.where(w3 > 1e-8, w3, 1e-8)

    w_orb = jnp.where(degenerate, w2, w3c * w123 / w13_safe)
    inc = jnp.where(degenerate, 0.0, jnp.arccos(
        jnp.clip(w2 * w3c / w13_safe / w123_safe, -1.0, 1.0)
    ))
    phi0 = jnp.where(degenerate, 0.0, jnp.arctan2(-w1 * w123, w3c * w13_safe))

    cphi0 = jnp.cos(phi0)
    sphi0 = jnp.sin(phi0)
    cinc = jnp.cos(inc)
    sinc = jnp.sin(inc)

    den0 = jnp.sqrt(cphi0 * cphi0 + cinc * cinc * sphi0 * sphi0)
    s_true = s / den0
    c_om = (cphi0 * calpha + cinc * salpha * sphi0) / den0
    s_om = (cphi0 * salpha - cinc * calpha * sphi0) / den0

    dt = times - t0
    phi = dt * w_orb + phi0
    cphi = jnp.cos(phi)
    sphi = jnp.sin(phi)
    den = jnp.sqrt(cphi * cphi + cinc * cinc * sphi * sphi)
    sep = s_true * den

    u = u0 + pai1 * et[:, 1] - pai2 * et[:, 0]
    tn = dt * tE_inv + pai1 * et[:, 0] + pai2 * et[:, 1]

    y1 = (cphi * (u * s_om - tn * c_om) + cinc * sphi * (u * c_om + tn * s_om)) / den
    y2 = (-cphi * (u * c_om + tn * s_om) - cinc * sphi * (tn * c_om - u * s_om)) / den
    _ = sinc  # kept for parity with the C++ variable list
    return y1, y2, sep


def kepler_trajectory(times, params, et, *, newton_iters=15):
    """Port of ``BinaryAstroLightCurveKepler``.

    ``params`` = ``[log_s, log_q, u0, alpha_0, log_rho, log_tE, t0, pai1, pai2,
    w1, w2, w3, szs, ar]``.  Returns ``(y1, y2, sep)``.
    """
    (
        log_s, _log_q, u0, alpha, _log_rho, log_tE, t0, pai1, pai2,
        w1, w2, w3, szs, ar_raw,
    ) = (params[i] for i in range(14))
    s = jnp.exp(log_s)
    tE_inv = jnp.exp(-log_tE)
    ar = ar_raw + 1e-8

    smix = 1.0 + szs * szs
    sqsmix = jnp.sqrt(smix)
    wt2 = w1 * w1 + w2 * w2 + w3 * w3
    arm2 = 2.0 * ar - 1.0
    n = jnp.sqrt(wt2 / arm2 / smix) / ar

    z = jnp.array([-szs * w2, szs * w1 - w3, w2])
    z = z / jnp.linalg.norm(z)

    x = jnp.array([
        -ar * w1 * w1 + (ar - 1.0) * w2 * w2 - arm2 * szs * w1 * w3
        + (ar - 1.0) * w3 * w3,
        -arm2 * w2 * (w1 + szs * w3),
        (ar - 1.0) * szs * (w1 * w1 + w2 * w2) - arm2 * w1 * w3
        - ar * szs * w3 * w3,
    ])
    e = jnp.linalg.norm(x)
    x = x / e
    e = e / (ar * sqsmix * wt2)
    y = jnp.cross(z, x)

    conu = (x[0] + x[2] * szs) / sqsmix
    cos_e0 = (conu + e) / (1.0 + e * conu)
    ee0 = jnp.arccos(jnp.clip(cos_e0, -1.0, 1.0))
    snu = y[0] + y[2] * szs
    sign = jnp.where(snu > 0.0, 1.0, -1.0)
    ee0 = ee0 * sign
    sin_e0 = jnp.sqrt(jnp.clip(1.0 - cos_e0 * cos_e0, 0.0, 1.0)) * sign
    tperi = t0 - (ee0 - e * sin_e0) / n

    a = ar * s * jnp.sqrt(smix)

    m = _wrap_pi(n * (times - tperi))

    def newton(_, ee):
        dm = m - (ee - e * jnp.sin(ee))
        ee = ee + dm / (1.0 - e * jnp.cos(ee))
        return jnp.clip(ee, -jnp.pi, jnp.pi)

    ee = jax.lax.fori_loop(0, newton_iters, newton, m + e * jnp.sin(m))

    r0 = a * (jnp.cos(ee) - e)
    r1 = a * jnp.sqrt(jnp.clip(1.0 - e * e, 0.0, None)) * jnp.sin(ee)
    px = r0 * x[0] + r1 * y[0]
    py = r0 * x[1] + r1 * y[1]
    sep = jnp.sqrt(px * px + py * py)
    psi = jnp.arctan2(py, px)

    dt = times - t0
    u = u0 + pai1 * et[:, 1] - pai2 * et[:, 0]
    tn = dt * tE_inv + pai1 * et[:, 0] + pai2 * et[:, 1]
    ang = alpha + psi
    y1 = -tn * jnp.cos(ang) + u * jnp.sin(ang)
    y2 = -u * jnp.cos(ang) - tn * jnp.sin(ang)
    return y1, y2, sep
