"""Single-lens JAX physics; annual trajectories supplied by lcbinint.

lcbinint supports binary/triple lenses, so PSPL is analytic and uniform FSPL
uses this package's image-contour integration. No VBM tables or runtime calls.
"""
import jax
import jax.numpy as jnp
import lcbinint
from .jax_fspl_contour import fspl_magnification

jax.config.update('jax_enable_x64', True)


class PointLens:
    def __init__(self):
        self.Tol = self.RelTol = 1e-4
        self._t0_par = None
        self.t0_par_fixed = 0
        self.sky = None
        self._curve = None

    @property
    def t0_par(self):
        return self._t0_par

    @t0_par.setter
    def t0_par(self, value):
        self._t0_par = value
        self._prepare_parallax()

    def _prepare_parallax(self):
        # lcbinint caches ephemeris arrays. Construct outside jit/grad tracing.
        if self.sky is not None and self._t0_par is not None:
            self._curve = lcbinint.LightCurve(
                model=lcbinint.Model(parallax=True, finite_source=False,
                                    sky=self.sky, t_ref=self._t0_par),
                options=lcbinint.Options(jax=True, coordinates='vbm_center_of_mass'))

    def SetObjectCoordinates(self, coordinates):
        ra, dec = coordinates.split()
        self.sky = lcbinint.obs.SkyCoord(ra, dec)
        self._prepare_parallax()

    def _evaluate(self, p, times, *, finite=False, parallax=False):
        times = jnp.asarray(times)
        u0 = p[0] if parallax else jnp.exp(p[0])
        tE = jnp.exp(p[1])
        if parallax:
            if self.sky is None or self.t0_par is None:
                raise ValueError('annual parallax requires sky and fixed reference epoch')
            if self._curve is None:
                raise ValueError('prepare the annual curve outside JAX tracing')
            index = 4 if finite else 3
            params = dict(t0=p[2], tE=tE, u0=u0, alpha=0., s=1.2, q=.001,
                          piEN=p[index], piEE=p[index+1])
            trajectory = self._curve.source_trajectory(times, params)
            x, y = trajectory.x, trajectory.y
        else:
            x = -(times-p[2])/tE
            y = jnp.full_like(times, -u0)
        u = jnp.sqrt(x*x+y*y)
        def pspl(value):
            return (value*value+2.)/(value*jnp.sqrt(value*value+4.))
        if finite:
            rho = jnp.exp(p[3])
            def at_point(value):
                # The far-field finite-source correction is below 1e-6 here.
                def contour(v):
                    result = fspl_magnification(v, rho, coarse_order=2048, fine_order=4096,
                        absolute_tolerance=self.Tol, relative_tolerance=self.RelTol)
                    return jnp.where(result.converged, result.magnification, jnp.nan)
                return jax.lax.cond(rho < 1e-3*value, pspl, contour, value)
            mag = jax.vmap(at_point)(u)
        else:
            mag = pspl(u)
        return mag, x, y

    def PSPLLightCurve(self, p, times):
        return self._evaluate(p, times)

    def ESPLLightCurve(self, p, times):
        return self._evaluate(p, times, finite=True)

    def PSPLLightCurveParallax(self, p, times):
        return self._evaluate(p, times, parallax=True)

    def ESPLLightCurveParallax(self, p, times):
        return self._evaluate(p, times, finite=True, parallax=True)
