"""``AutodiffVBM`` -- a differentiable drop-in for ``VBMicrolensing``.

The class mirrors ``VBMicrolensing.VBMicrolensing``: create an instance, set
``Tol``/``RelTol`` (and coordinates for parallax), then call the same methods
with the same parameter conventions.  Every result is a JAX array with an
analytic derivative, so the light curves compose directly into ``jax.grad`` /
``jax.jit`` / a numpyro model.

Supported (binary lens, uniform-brightness source):

* ``BinaryMag0``, ``BinaryMag2``, ``BinaryMag``
* ``BinaryLightCurve``, ``BinaryLightCurveParallax``,
  ``BinaryLightCurveOrbital``, ``BinaryLightCurveKepler``
* ``PSPLLightCurve``, ``PSPLLightCurveParallax``, ``PSPLMag``
* ``ESPLMag``, ``ESPLMag2``, ``ESPLLightCurve``, ``ESPLLightCurveParallax``

Two extra attributes, absent from VBMicrolensing, tune the HMC behaviour:

* ``finite_mask`` -- fixed per-epoch Boolean array.  ``True`` forces the
  finite-source contour branch, ``False`` the analytic point-source branch.
  A fixed mask keeps ``BinaryMag2``'s parameter-dependent point/finite switch
  from putting a small discontinuity in the sampled target.  ``None`` keeps
  VBM's adaptive choice.
* ``grad_tol`` -- absolute per-epoch Jacobian-error budget, decoupled from
  ``Tol``/``RelTol`` (see :func:`autodiff_vbm._kernel.binary_mag_from_trajectory`).
* ``strict_grad_gate`` -- ``False`` (default): an epoch that misses ``grad_tol``
  returns a best-effort finite Jacobian row.  ``True``: NaN row (offline
  diagnostics only).
* ``espl_table_only`` -- force ``ESPLMag2``'s table branch everywhere instead of
  its parameter-dependent switch to the closed-form quadrupole expansion.
"""

from __future__ import annotations

import jax.numpy as jnp

from ._ephemeris import earth_projected_offsets, parse_coordinate_string
from ._espl import espl_magnification
from ._kernel import binary_mag_from_trajectory, binary_mag_gradient_report
from ._orbital import circular_trajectory, kepler_trajectory


class AutodiffVBM:
    def __init__(self):
        self.Tol = 1e-2
        self.RelTol = 1e-3
        self.t0_par_fixed = 0
        self.t0_par = 7000.0
        self.parallaxsystem = 1
        self.t_in_HJD = True
        # HMC-only knobs (not present in VBMicrolensing).
        self.finite_mask = None
        self.grad_tol = None
        # False (default): uncertified Jacobian rows fall back to a best-effort
        # finite derivative.  True restores NaN rows (offline diagnostics only).
        self.strict_grad_gate = False
        self.espl_table_only = False
        self._ra = None
        self._dec = None
        self._coordinates_set = False

    # ------------------------------------------------------------------
    # Coordinates
    # ------------------------------------------------------------------
    def SetObjectCoordinates(self, coordinate_string, satellite_dir=None):
        """Set the target coordinates from ``"hr:mn:sc deg:pr:sc"``.

        ``satellite_dir`` is accepted for signature compatibility with
        VBMicrolensing but satellite parallax is not implemented.
        """
        if satellite_dir is not None:
            raise NotImplementedError("satellite parallax is not implemented")
        self._ra, self._dec = parse_coordinate_string(coordinate_string)
        self._coordinates_set = True

    def _require_coordinates(self):
        if not self._coordinates_set:
            raise RuntimeError(
                "call SetObjectCoordinates before a parallax light curve"
            )

    # ------------------------------------------------------------------
    # Magnification primitives
    # ------------------------------------------------------------------
    def _mag(self, s, q, y1, y2, rho, *, branch, tol, retol):
        y1 = jnp.atleast_1d(jnp.asarray(y1, dtype=jnp.float64))
        y2 = jnp.atleast_1d(jnp.asarray(y2, dtype=jnp.float64))
        sep = jnp.broadcast_to(jnp.asarray(s, dtype=jnp.float64), y1.shape)
        if branch is None:
            mask = None
        else:
            mask = jnp.full(y1.shape, bool(branch))
        mag = binary_mag_from_trajectory(
            y1, y2, rho, q, sep,
            tol=tol, retol=retol,
            finite_mask=mask, grad_tol=self.grad_tol,
            strict_grad_gate=self.strict_grad_gate,
        )
        return mag

    def BinaryMag0(self, s, q, y1, y2):
        """Point-source magnification by a binary lens."""
        out = self._mag(s, q, y1, y2, 1e-10, branch=False,
                        tol=self.Tol, retol=self.RelTol)
        return out if out.shape != (1,) else out[0]

    def BinaryMag2(self, s, q, y1, y2, rho):
        """Adaptive finite-source magnification (VBM's point/finite switch)."""
        out = self._mag(s, q, y1, y2, rho, branch=None,
                        tol=self.Tol, retol=self.RelTol)
        return out if out.shape != (1,) else out[0]

    def BinaryMag(self, s, q, y1, y2, rho, accuracy=None):
        """Finite-source contour magnification to an absolute ``accuracy``."""
        tol = self.Tol if accuracy is None else accuracy
        out = self._mag(s, q, y1, y2, rho, branch=True,
                        tol=tol, retol=self.RelTol)
        return out if out.shape != (1,) else out[0]

    @staticmethod
    def PSPLMag(u):
        """Paczynski point-source point-lens magnification."""
        u2 = jnp.asarray(u, dtype=jnp.float64) ** 2
        return (u2 + 2.0) / jnp.sqrt(u2 * (u2 + 4.0))

    # ------------------------------------------------------------------
    # Binary-lens light curves
    # ------------------------------------------------------------------
    def _binary_curve(self, y1, y2, sep, rho, q):
        return binary_mag_from_trajectory(
            y1, y2, rho, q, sep,
            tol=self.Tol, retol=self.RelTol,
            finite_mask=self.finite_mask, grad_tol=self.grad_tol,
            strict_grad_gate=self.strict_grad_gate,
        )

    def BinaryLightCurve(self, params, times):
        """params = [log_s, log_q, u0, alpha, log_rho, log_tE, t0]."""
        p = jnp.asarray(params, dtype=jnp.float64)
        times = jnp.asarray(times, dtype=jnp.float64)
        s = jnp.exp(p[0])
        q = jnp.exp(p[1])
        rho = jnp.exp(p[4])
        tE_inv = jnp.exp(-p[5])
        salpha = jnp.sin(p[3])
        calpha = jnp.cos(p[3])
        tn = (times - p[6]) * tE_inv
        y1 = p[2] * salpha - tn * calpha
        y2 = -p[2] * calpha - tn * salpha
        sep = jnp.broadcast_to(s, times.shape)
        mag = self._binary_curve(y1, y2, sep, rho, q)
        return [mag, y1, y2]

    def _parallax_uv(self, params, times, t0):
        self._require_coordinates()
        t0_par = t0 if self.t0_par_fixed == 0 else jnp.asarray(self.t0_par)
        et = earth_projected_offsets(times, t0_par, self._ra, self._dec)
        return et

    def BinaryLightCurveParallax(self, params, times):
        """params = [log_s, log_q, u0, alpha, log_rho, log_tE, t0, pai1, pai2].

        ``pai1``/``pai2`` are the parallax components parallel and orthogonal to
        Earth's acceleration, as in VBMicrolensing.
        """
        p = jnp.asarray(params, dtype=jnp.float64)
        times = jnp.asarray(times, dtype=jnp.float64)
        s = jnp.exp(p[0])
        q = jnp.exp(p[1])
        rho = jnp.exp(p[4])
        tE_inv = jnp.exp(-p[5])
        salpha = jnp.sin(p[3])
        calpha = jnp.cos(p[3])
        et = self._parallax_uv(p, times, p[6])
        tn = (times - p[6]) * tE_inv + p[7] * et[:, 0] + p[8] * et[:, 1]
        u = p[2] + p[7] * et[:, 1] - p[8] * et[:, 0]
        y1 = u * salpha - tn * calpha
        y2 = -u * calpha - tn * salpha
        sep = jnp.broadcast_to(s, times.shape)
        mag = self._binary_curve(y1, y2, sep, rho, q)
        return [mag, y1, y2]

    def BinaryLightCurveOrbital(self, params, times):
        """params = [log_s, log_q, u0, alpha_0, log_rho, log_tE, t0, pai1, pai2,
        w1, w2, w3] with w1=(ds/dt)/s, w2=dalpha/dt, w3=(dsz/dt)/s."""
        p = jnp.asarray(params, dtype=jnp.float64)
        times = jnp.asarray(times, dtype=jnp.float64)
        q = jnp.exp(p[1])
        rho = jnp.exp(p[4])
        et = self._parallax_uv(p, times, p[6])
        y1, y2, sep = circular_trajectory(times, p, et)
        mag = self._binary_curve(y1, y2, sep, rho, q)
        return [mag, y1, y2, sep]

    def BinaryLightCurveKepler(self, params, times):
        """params = [log_s, log_q, u0, alpha_0, log_rho, log_tE, t0, pai1, pai2,
        w1, w2, w3, szs, ar] (full Keplerian orbital motion)."""
        p = jnp.asarray(params, dtype=jnp.float64)
        times = jnp.asarray(times, dtype=jnp.float64)
        q = jnp.exp(p[1])
        rho = jnp.exp(p[4])
        et = self._parallax_uv(p, times, p[6])
        y1, y2, sep = kepler_trajectory(times, p, et)
        mag = self._binary_curve(y1, y2, sep, rho, q)
        return [mag, y1, y2, sep]

    # ------------------------------------------------------------------
    # PSPL light curves
    # ------------------------------------------------------------------
    def PSPLLightCurve(self, params, times):
        """params = [log_u0, log_tE, t0]."""
        p = jnp.asarray(params, dtype=jnp.float64)
        times = jnp.asarray(times, dtype=jnp.float64)
        u0 = jnp.exp(p[0])
        tE_inv = jnp.exp(-p[1])
        tn = (times - p[2]) * tE_inv
        u2 = tn * tn + u0 * u0
        mag = (u2 + 2.0) / jnp.sqrt(u2 * (u2 + 4.0))
        return [mag, -tn, jnp.broadcast_to(-u0, times.shape)]

    def PSPLLightCurveParallax(self, params, times):
        """params = [u0, log_tE, t0, pai1, pai2] (note: linear ``u0``)."""
        p = jnp.asarray(params, dtype=jnp.float64)
        times = jnp.asarray(times, dtype=jnp.float64)
        tE_inv = jnp.exp(-p[1])
        et = self._parallax_uv(p, times, p[2])
        tn = (times - p[2]) * tE_inv + p[3] * et[:, 0] + p[4] * et[:, 1]
        u1 = p[0] + p[3] * et[:, 1] - p[4] * et[:, 0]
        u2 = tn * tn + u1 * u1
        mag = (u2 + 2.0) / jnp.sqrt(u2 * (u2 + 4.0))
        return [mag, -tn, -u1]

    # ------------------------------------------------------------------
    # ESPL (extended-source point-lens) -- uniform source
    # ------------------------------------------------------------------
    def ESPLMag(self, u, rho):
        """Table-lookup finite-source magnification for a point lens.

        Matches ``VBMicrolensing.ESPLMag`` (the raw table branch, without
        ``ESPLMag2``'s closed-form shortcut for small ``rho``).
        """
        return espl_magnification(u, rho, tol=self.Tol, table_only=True)

    def ESPLMag2(self, u, rho):
        """Finite-source point-lens magnification, as ``VBMicrolensing.ESPLMag2``.

        Uses the closed-form quadrupole expansion where it is accurate to
        ``Tol`` and the table lookup elsewhere, unless ``espl_table_only`` is
        set.
        """
        return espl_magnification(
            u, rho, tol=self.Tol, table_only=self.espl_table_only
        )

    def ESPLLightCurve(self, params, times):
        """params = [log_u0, log_tE, t0, log_rho]."""
        p = jnp.asarray(params, dtype=jnp.float64)
        times = jnp.asarray(times, dtype=jnp.float64)
        u0 = jnp.exp(p[0])
        tE_inv = jnp.exp(-p[1])
        rho = jnp.exp(p[3])
        tn = (times - p[2]) * tE_inv
        u = jnp.sqrt(tn * tn + u0 * u0)
        mag = espl_magnification(
            u, rho, tol=self.Tol, table_only=self.espl_table_only
        )
        return [mag, -tn, jnp.broadcast_to(-u0, times.shape)]

    def ESPLLightCurveParallax(self, params, times):
        """params = [u0, log_tE, t0, log_rho, pai1, pai2] (note: linear ``u0``)."""
        p = jnp.asarray(params, dtype=jnp.float64)
        times = jnp.asarray(times, dtype=jnp.float64)
        tE_inv = jnp.exp(-p[1])
        rho = jnp.exp(p[3])
        et = self._parallax_uv(p, times, p[2])
        tn = (times - p[2]) * tE_inv + p[4] * et[:, 0] + p[5] * et[:, 1]
        u1 = p[0] + p[4] * et[:, 1] - p[5] * et[:, 0]
        u = jnp.sqrt(tn * tn + u1 * u1)
        mag = espl_magnification(
            u, rho, tol=self.Tol, table_only=self.espl_table_only
        )
        return [mag, -tn, -u1]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def gradient_report(self, params, times, kind="binary"):
        """Non-differentiable ``(mag, jacobian, gradient_error)`` for a curve.

        ``kind`` is ``"binary"`` (linear) or ``"parallax"``.  See
        :func:`autodiff_vbm._kernel.binary_mag_gradient_report`.
        """
        p = jnp.asarray(params, dtype=jnp.float64)
        times = jnp.asarray(times, dtype=jnp.float64)
        s = jnp.exp(p[0])
        q = jnp.exp(p[1])
        rho = jnp.exp(p[4])
        tE_inv = jnp.exp(-p[5])
        salpha = jnp.sin(p[3])
        calpha = jnp.cos(p[3])
        if kind == "parallax":
            et = self._parallax_uv(p, times, p[6])
            tn = (times - p[6]) * tE_inv + p[7] * et[:, 0] + p[8] * et[:, 1]
            u = p[2] + p[7] * et[:, 1] - p[8] * et[:, 0]
        else:
            tn = (times - p[6]) * tE_inv
            u = p[2]
        y1 = u * salpha - tn * calpha
        y2 = -u * calpha - tn * salpha
        sep = jnp.broadcast_to(s, times.shape)
        return binary_mag_gradient_report(
            y1, y2, rho, q, sep,
            tol=self.Tol, retol=self.RelTol,
            finite_mask=self.finite_mask, grad_tol=self.grad_tol,
            strict_grad_gate=self.strict_grad_gate,
        )
