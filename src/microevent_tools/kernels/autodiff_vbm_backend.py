"""Small lcbinint-compatible facade backed by the local autodiff-vbm tree."""

from __future__ import annotations

# Support both installed modules and the internal subprocess entry points.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "microevent_tools.kernels"


from dataclasses import dataclass
from types import SimpleNamespace

import jax.numpy as jnp
from autodiff_vbm import AutodiffVBM


class Options:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@dataclass
class SkyCoord:
    ra: str
    dec: str


obs = SimpleNamespace(SkyCoord=SkyCoord)


class Model:
    def __init__(self, parallax=False, sky=None, t_ref=None, **kwargs):
        self.parallax = parallax
        self.sky = sky
        self.t_ref = t_ref


class _WarmupReport:
    all_calibrated = True


class LightCurve:
    def __init__(self, model=None, options=None):
        self.model = model or Model()
        self.options = options or Options()
        self.vbm = AutodiffVBM()
        self.vbm.Tol = self.options.kwargs.get("tol", 1e-4)
        self.vbm.RelTol = self.options.kwargs.get("reltol", 1e-4)
        if self.model.parallax:
            if self.model.sky is None:
                raise ValueError("parallax model requires sky coordinates")
            self.vbm.SetObjectCoordinates(f"{self.model.sky.ra} {self.model.sky.dec}")
            if self.model.t_ref is not None:
                self.vbm.t0_par_fixed = 1
                self.vbm.t0_par = float(self.model.t_ref)

    def warmup(self, times, parameters):
        # autodiff-vbm has no trace-compilation warmup plan.  Evaluate once so
        # native-extension/configuration failures occur before optimization.
        self(jnp.asarray(times), parameters)
        return _WarmupReport()

    def __call__(self, times, parameters):
        p = [
            jnp.log(parameters["s"]), jnp.log(parameters["q"]),
            parameters["u0"], parameters["alpha"], jnp.log(parameters["rho"]),
            jnp.log(parameters["tE"]), parameters["t0"],
        ]
        if self.model.parallax:
            # These are VBM pai1/pai2 components, not north/east components.
            p += [parameters.get("pai1", 0.0), parameters.get("pai2", 0.0)]
            return self.vbm.BinaryLightCurveParallax(jnp.asarray(p), times)[0]
        return self.vbm.BinaryLightCurve(jnp.asarray(p), times)[0]
