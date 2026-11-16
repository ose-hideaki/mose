"""Differentiable VBMicrolensing.

``autodiff_vbm`` mirrors the ``VBMicrolensing`` Python API -- same class, same
method names, same parameter conventions -- but every result is a JAX array
carrying an analytic derivative, so a light curve drops straight into
``jax.grad`` / ``jax.jit`` / a numpyro model.

Quick start::

    import jax
    from autodiff_vbm import AutodiffVBM

    jax.config.update("jax_enable_x64", True)

    vbm = AutodiffVBM()
    vbm.Tol = 1e-3
    mag, y1, y2 = vbm.BinaryLightCurve(params, times)

The finite-source magnification is evaluated by the unmodified VBMicrolensing
5.5 contour engine through a fused CPU FFI; the orbital models (annual
parallax, circular and Keplerian lens orbital motion) are pure JAX.
"""

from .core import AutodiffVBM
from ._espl import espl_magnification
from ._kernel import (
    binary_mag_from_trajectory,
    binary_mag_gradient_report,
    set_grad_gate_report,
)

__all__ = [
    "AutodiffVBM",
    "binary_mag_from_trajectory",
    "binary_mag_gradient_report",
    "set_grad_gate_report",
    "espl_magnification",
]

__version__ = "0.1.0"
