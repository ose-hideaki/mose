"""Fused VBMicrolensing CPU FFI with a custom JVP.

The C++ handler (``native/vbm_ffi.cpp``) statically links the unmodified
VBMicrolensing 5.5 contour engine and evaluates a whole light curve -- root
solving, point/finite-source routing, adaptive contour integration -- in one
call.  A separate value/Jacobian handler feeds a hand-rolled forward-mode dual
kernel through :func:`jax.custom_jvp`, and the derivative is analytic.

Value/derivative consistency (the HMC-critical property): for the deterministic
branches (``0``/``1``) the returned magnification is *not* VBM's own adaptively
refined ``BinaryMag``/``BinaryMag0`` return.  It is the magnification
reconstructed from VBM's frozen image support by the same area walk that the
analytic Jacobian differentiates, so ``value`` and ``d(value)/d(theta)`` are the
primal and tangent of one effective function -- as in a pure-JAX target.  The
forward and JVP handlers share that reconstruction bit-for-bit.  Parity with
VBM's refined value is then ~1e-5..1e-6 at ``tol=1e-3`` (a frozen-support area
walk, not an adaptive one); tighten ``tol``/``retol`` for a closer match.  The
adaptive branch (``-1``) still returns VBM's ``BinaryMag2`` value directly and
is not recommended for HMC.

Per-epoch operands (all shape ``(epochs,)`` unless noted):

* ``source_x``, ``source_y`` -- source position in the binary centre-of-mass
  frame, matching VBMicrolensing's ``BinaryMag2(s, q, y1, y2, rho)``;
* ``rho``, ``q`` -- scalar source radius and mass ratio;
* ``separation`` -- projected lens separation, per epoch (orbital motion);
* ``branch_mode`` -- ``-1`` VBM adaptive, ``0`` force point source,
  ``1`` force finite source.  HMC uses a fixed (parameter-independent) mask;
* ``tol``, ``retol`` -- VBM value-accuracy goals;
* ``grad_tol`` -- absolute per-epoch Jacobian-error budget, decoupled from
  ``tol``/``retol``.  An epoch the analytic two-form estimator cannot certify
  within this budget still returns its best-effort finite Jacobian row by
  default (a NaN row would poison an HMC leapfrog step); pass
  ``strict_grad_gate=True`` for the honest-refusal NaN row instead.
"""

from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp

try:
    # JAX 0.4.x exposes the FFI API from jax.extend.  Newer releases may also
    # re-export it as jax.ffi, but using the original location keeps the
    # declared minimum version (0.4.31) working.
    from jax.extend import ffi as jax_ffi
except ImportError:  # pragma: no cover - compatibility with future layouts
    from jax import ffi as jax_ffi


_FORWARD_TARGET = "autodiff_vbm_batch_f64_v1"
_JACOBIAN_TARGET = "autodiff_vbm_batch_jacobian_f64_v1"


# Optional, default-off notification when a finite-source epoch's Jacobian row
# could not be certified within ``grad_tol`` and a best-effort derivative was
# handed back instead.  ``"off"`` keeps the callback out of the compiled graph
# entirely (no per-step host round-trip, no risk of the microlux-style
# compile-time blow-up from an in-loop warning).  ``"summary"`` installs ONE
# ``jax.debug.callback`` on the whole per-curve gradient-error vector, after the
# single batched FFI call -- never inside an epoch/vmap/scan loop.
_GRAD_GATE_REPORT = "off"


def set_grad_gate_report(mode: str) -> None:
    """Set the best-effort-Jacobian notification mode: ``"off"`` or ``"summary"``.

    Takes effect on the next trace/compile.  ``"off"`` is the default and is the
    right choice for production HMC.
    """

    global _GRAD_GATE_REPORT
    if mode not in ("off", "summary"):
        raise ValueError('grad-gate report mode must be "off" or "summary"')
    _GRAD_GATE_REPORT = mode


def _report_grad_gate(gradient_error, grad_tol):
    if _GRAD_GATE_REPORT != "summary":
        return

    def _cb(errors, budget):
        import numpy as _np

        budget = float(budget)
        if not (budget > 0.0):
            return
        errors = _np.asarray(errors)
        over = errors > budget
        count = int(over.sum())
        if count:
            worst = float(errors[over].max())
            print(
                f"[autodiff-vbm] {count} epoch(s) exceeded grad_tol={budget:.2e} "
                f"(worst {worst:.2e}); best-effort Jacobian returned.",
                flush=True,
            )

    jax.debug.callback(_cb, gradient_error, grad_tol)


@lru_cache(maxsize=1)
def _register_targets() -> None:
    try:
        from autodiff_vbm import _autodiff_vbm_native
    except ImportError as error:
        raise RuntimeError(
            "the autodiff-vbm native extension is not built; from an editable "
            "checkout run `autodiff-vbm-build-native` "
            "(or `python -m autodiff_vbm._native_build`)"
        ) from error

    jax_ffi.register_ffi_target(
        _FORWARD_TARGET,
        _autodiff_vbm_native.vbm_batch_ffi(),
        platform="cpu",
    )
    jax_ffi.register_ffi_target(
        _JACOBIAN_TARGET,
        _autodiff_vbm_native.vbm_batch_jacobian_ffi(),
        platform="cpu",
    )


def _ffi_call(inputs, *, include_jacobian):
    epochs = inputs[0].shape[0]
    if include_jacobian:
        outputs = (
            jax.ShapeDtypeStruct((epochs,), jnp.float64),
            jax.ShapeDtypeStruct((epochs, 5), jnp.float64),
            jax.ShapeDtypeStruct((epochs,), jnp.float64),
        )
        return jax_ffi.ffi_call(_JACOBIAN_TARGET, outputs, *inputs)
    outputs = (jax.ShapeDtypeStruct((epochs,), jnp.float64),)
    # The forward handler has no use for the gradient-budget knob or the strict
    # gate; keep only the first 8 operands.
    return jax_ffi.ffi_call(_FORWARD_TARGET, outputs, *inputs[:8])


@jax.custom_jvp
def _vbm_kernel(
    source_x, source_y, rho, q, s, branch_mode, tol, retol, grad_tol,
    strict_grad_gate,
):
    inputs = (
        source_x, source_y, rho, q, s, branch_mode, tol, retol, grad_tol,
        strict_grad_gate,
    )
    return _ffi_call(inputs, include_jacobian=False)[0]


@_vbm_kernel.defjvp
def _vbm_kernel_jvp(primals, tangents):
    (source_x, source_y, rho, q, s, branch_mode, tol, retol, grad_tol,
     strict_grad_gate) = primals
    inputs = (
        source_x, source_y, rho, q, s, branch_mode, tol, retol, grad_tol,
        strict_grad_gate,
    )
    magnification, jacobian, gradient_error = _ffi_call(
        inputs, include_jacobian=True
    )
    _report_grad_gate(gradient_error, grad_tol)
    tangent = (
        jacobian[:, 0] * tangents[0]
        + jacobian[:, 1] * tangents[1]
        + jacobian[:, 2] * tangents[2]
        + jacobian[:, 3] * tangents[3]
        + jacobian[:, 4] * tangents[4]
    )
    return magnification, tangent


def _prepare_ffi_inputs(
    source_x, source_y, rho, q, separation, tol, retol, grad_tol, finite_mask,
    strict_grad_gate=False,
):
    """Validate, canonicalize, and pack the per-epoch native FFI operands."""

    _register_targets()
    if jax.default_backend() != "cpu":
        raise RuntimeError("the VBM FFI backend is CPU-only")
    source_x = jnp.asarray(source_x, dtype=jnp.float64)
    source_y = jnp.asarray(source_y, dtype=jnp.float64)
    separation = jnp.asarray(separation, dtype=jnp.float64)
    if source_x.ndim != 1 or source_y.ndim != 1 or separation.ndim != 1:
        raise ValueError("source_x, source_y, and separation must be one-dimensional")
    if source_y.shape != source_x.shape or separation.shape != source_x.shape:
        raise ValueError("source_x, source_y, and separation must have equal length")
    if finite_mask is None:
        # -1 preserves VBMicrolensing's historical BinaryMag2 adaptive
        # point/finite-source choice.  0 and 1 are deterministic choices used
        # by HMC, where a parameter-dependent branch switch would make the
        # target discontinuous.
        branch_mode = jnp.full(source_x.shape, -1, dtype=jnp.int32)
    else:
        finite_mask = jnp.asarray(finite_mask)
        if finite_mask.ndim != 1 or finite_mask.shape != source_x.shape:
            raise ValueError("finite_mask must be one-dimensional and match source_x")
        branch_mode = jnp.where(
            finite_mask.astype(bool),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )
    if grad_tol is None:
        # A non-positive gradient tolerance tells the native handler to fall
        # back to its value-tied heuristic.
        grad_tol = 0.0
    rho, q, tol, retol, grad_tol = (
        jnp.asarray(value, dtype=jnp.float64)
        for value in (rho, q, tol, retol, grad_tol)
    )
    # Default False: an uncertified epoch returns its best-effort finite
    # Jacobian row.  True restores the honest-refusal NaN row (offline
    # diagnostics only -- a NaN row poisons an HMC leapfrog step).
    strict_flag = jnp.asarray(bool(strict_grad_gate), dtype=jnp.int32)

    # VBM canonicalizes q > 1 by exchanging the lens labels.  Reflecting the
    # source x coordinate is the corresponding symmetry in this frame.
    swapped = q > 1.0
    source_x = jnp.where(swapped, -source_x, source_x)
    q = jnp.where(swapped, 1.0 / q, q)
    return (
        source_x, source_y, rho, q, separation, branch_mode, tol, retol,
        grad_tol, strict_flag,
    )


def binary_mag_from_trajectory(
    source_x,
    source_y,
    rho,
    q,
    separation,
    tol=1e-2,
    retol=1e-3,
    *,
    finite_mask=None,
    grad_tol=None,
    strict_grad_gate=False,
):
    """Evaluate a curve from per-epoch centre-of-mass source coordinates.

    ``separation`` may vary per epoch, so this entry point can represent
    orbital motion while keeping the whole light curve inside one native call.

    ``grad_tol`` sets an absolute per-epoch Jacobian-accuracy budget decoupled
    from ``tol``/``retol``.  Each finite-source epoch's analytic contour
    Jacobian is checked with a two-form parabola-disagreement estimator (the
    derivative analogue of VBM's own value estimator -- no finite differencing).
    An ordinary epoch whose primal-support estimate already clears the budget
    is certified with no extra contour solve; otherwise it runs a short
    ``0.25x`` refinement loop.  A caustic-connection epoch runs a bounded
    coarse-to-fine sweep (floor ``1e-9``).  An epoch that cannot be certified
    within the budget still returns its best-effort finite Jacobian row (the
    last refinement iterate); set ``strict_grad_gate=True`` to get a ``NaN`` row
    instead (offline diagnostics only -- a NaN row poisons an HMC leapfrog
    step).  ``None`` for ``grad_tol`` selects a value-tied default of
    ``10 * tol`` clamped to ``[1e-4, 1e-1]``.  Enable a default-off summary of
    uncertified epochs with :func:`set_grad_gate_report`.
    """

    inputs = _prepare_ffi_inputs(
        source_x, source_y, rho, q, separation, tol, retol, grad_tol,
        finite_mask, strict_grad_gate,
    )
    return _vbm_kernel(*inputs)


def binary_mag_gradient_report(
    source_x,
    source_y,
    rho,
    q,
    separation,
    tol=1e-2,
    retol=1e-3,
    *,
    finite_mask=None,
    grad_tol=None,
    strict_grad_gate=False,
):
    """Return ``(magnification, jacobian, gradient_error)`` for diagnostics.

    ``jacobian`` has shape ``(epochs, 5)`` over
    ``(source_x, source_y, rho, q, separation)``.  ``gradient_error[i]`` is the
    achieved Jacobian-error estimate for finite-source epoch ``i`` and ``0`` for
    analytic point-source epochs.  By default the matching ``jacobian`` row is a
    best-effort finite derivative even when ``gradient_error[i] > grad_tol``;
    pass ``strict_grad_gate=True`` for a ``NaN`` row on uncertified epochs.
    Filter on ``gradient_error > grad_tol`` for the pull-model notification.
    This bypasses the custom JVP and is not differentiable.
    """

    inputs = _prepare_ffi_inputs(
        source_x, source_y, rho, q, separation, tol, retol, grad_tol,
        finite_mask, strict_grad_gate,
    )
    return _ffi_call(inputs, include_jacobian=True)
