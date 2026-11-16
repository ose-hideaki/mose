"""Differentiable single-lens FSPL magnification from analytic image contours.

For every point on a uniform circular source boundary, the two point-lens
images are known analytically.  Their signed areas are evaluated with Green's
theorem.  The total magnification is

    (signed_area_positive - signed_area_negative) / (pi * rho**2).

This expression works both when the lens is inside and outside the source.
Midpoint angular nodes avoid evaluating the singular source point exactly at
u == rho.  Static array shapes make the kernel compatible with jit/vmap/grad.
"""

from __future__ import annotations

# Support both installed modules and the internal subprocess entry points.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "microevent_tools.kernels"


from typing import NamedTuple

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class FSPLContourResult(NamedTuple):
    magnification: jax.Array
    coarse_magnification: jax.Array
    estimated_error: jax.Array
    converged: jax.Array


def _signed_polygon_area(z: jax.Array) -> jax.Array:
    z_next = jnp.roll(z, -1)
    return 0.5 * jnp.sum(
        jnp.real(z) * jnp.imag(z_next) - jnp.imag(z) * jnp.real(z_next)
    )


def _magnification_at_order(u: jax.Array, rho: jax.Array, order: int) -> jax.Array:
    # The lens is at the origin and the source centre is on the positive real
    # axis. Rotational symmetry means no generality is lost.
    theta = 2.0 * jnp.pi * (jnp.arange(order, dtype=jnp.float64) + 0.5) / order
    source = u + rho * jnp.exp(1j * theta)
    radius = jnp.abs(source)
    direction = source / radius
    root = jnp.sqrt(radius * radius + 4.0)
    image_positive = 0.5 * (radius + root) * direction
    image_negative = 0.5 * (radius - root) * direction
    image_area = _signed_polygon_area(image_positive) - _signed_polygon_area(
        image_negative
    )
    return image_area / (jnp.pi * rho * rho)


def fspl_magnification(
    u: jax.Array,
    rho: jax.Array,
    *,
    coarse_order: int = 2048,
    fine_order: int = 4096,
    absolute_tolerance: float = 1.0e-5,
    relative_tolerance: float = 1.0e-5,
) -> FSPLContourResult:
    """Return uniform-source point-lens magnification and an error estimate."""
    if coarse_order < 8 or fine_order <= coarse_order:
        raise ValueError("require 8 <= coarse_order < fine_order")
    u = jnp.asarray(u, dtype=jnp.float64)
    rho = jnp.asarray(rho, dtype=jnp.float64)
    coarse = _magnification_at_order(u, rho, coarse_order)
    fine = _magnification_at_order(u, rho, fine_order)
    error = jnp.abs(fine - coarse)
    budget = absolute_tolerance + relative_tolerance * jnp.abs(fine)
    valid = jnp.isfinite(u) & jnp.isfinite(rho) & (u >= 0.0) & (rho > 0.0)
    return FSPLContourResult(
        jnp.where(valid, fine, jnp.nan),
        jnp.where(valid, coarse, jnp.nan),
        jnp.where(valid, error, jnp.nan),
        valid & jnp.isfinite(error) & (error <= budget),
    )


def fspl_magnification_value(u: jax.Array, rho: jax.Array, **kwargs) -> jax.Array:
    """Convenience scalar-returning wrapper for optimization and inference."""
    return fspl_magnification(u, rho, **kwargs).magnification


def fspl_magnification_batch(u: jax.Array, rho: jax.Array, **kwargs) -> jax.Array:
    """Vectorized magnification for an arbitrary array of lens separations."""
    values = jax.vmap(lambda value: fspl_magnification_value(value, rho, **kwargs))(
        jnp.ravel(jnp.asarray(u, dtype=jnp.float64))
    )
    return values.reshape(jnp.shape(u))


def fspl_light_curve(
    times: jax.Array,
    t0: jax.Array,
    tE: jax.Array,
    u0: jax.Array,
    rho: jax.Array,
    **kwargs,
) -> jax.Array:
    """Return a rectilinear, static single-lens finite-source light curve."""
    times = jnp.asarray(times, dtype=jnp.float64)
    tau = (times - t0) / tE
    separation = jnp.sqrt(u0 * u0 + tau * tau)
    return fspl_magnification_batch(separation, rho, **kwargs)


__all__ = [
    "FSPLContourResult",
    "fspl_magnification",
    "fspl_magnification_value",
    "fspl_magnification_batch",
    "fspl_light_curve",
]
