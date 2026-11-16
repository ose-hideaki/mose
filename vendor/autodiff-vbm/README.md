# autodiff-vbm

**Differentiable VBMicrolensing.** The same API as
[VBMicrolensing](https://github.com/valboz/VBMicrolensing) — same class, same
method names, same parameter conventions — returning JAX arrays with analytic
derivatives. A microlensing light curve drops straight into `jax.grad`,
`jax.jit`, or a numpyro model.

```python
import jax
from autodiff_vbm import AutodiffVBM

jax.config.update("jax_enable_x64", True)

vbm = AutodiffVBM()
vbm.Tol = 1e-3
vbm.RelTol = 1e-3

params = [                      # exactly VBMicrolensing's BinaryLightCurve layout
    jax.numpy.log(0.9),         # log s
    jax.numpy.log(1e-3),        # log q
    0.1,                        # u0
    1.3,                        # alpha  (radians)
    jax.numpy.log(0.01),        # log rho
    jax.numpy.log(40.0),        # log tE
    8280.0,                     # t0
]
times = jax.numpy.linspace(8260.0, 8300.0, 400)

mag, y1, y2 = vbm.BinaryLightCurve(params, times)

# ... and its gradient
grad = jax.grad(lambda p: vbm.BinaryLightCurve(p, times)[0].sum())(
    jax.numpy.asarray(params)
)
```

## What is reproduced

| VBMicrolensing method            | status | derivative |
|----------------------------------|--------|------------|
| `BinaryMag0`                     | ✅     | analytic (point-source, exact) |
| `BinaryMag2`                     | ✅     | analytic contour Jacobian |
| `BinaryMag`                      | ✅     | analytic contour Jacobian |
| `BinaryLightCurve`               | ✅     | analytic |
| `BinaryLightCurveParallax`       | ✅     | analytic (annual parallax, `pai1`/`pai2`) |
| `BinaryLightCurveOrbital`        | ✅     | analytic (circular lens orbital motion) |
| `BinaryLightCurveKepler`         | ✅     | analytic (full Keplerian orbital motion) |
| `PSPLLightCurve`                 | ✅     | analytic (pure JAX) |
| `PSPLLightCurveParallax`         | ✅     | analytic (pure JAX) |
| `PSPLMag`                        | ✅     | analytic (pure JAX) |
| `ESPLMag` / `ESPLMag2`           | ✅     | analytic (pure JAX, table lookup) |
| `ESPLLightCurve`                 | ✅     | analytic (pure JAX) |
| `ESPLLightCurveParallax`         | ✅     | analytic (pure JAX) |
| `SetObjectCoordinates`           | ✅     | — |

The finite-source magnification is evaluated by the **unmodified
VBMicrolensing 5.5 contour engine** through a fused CPU FFI, so the value is
VBMicrolensing-identical to numerical precision. The orbital models (parallax
via `SunEphemeris.txt` interpolation, circular and Keplerian lens orbital
motion) are pure JAX ports of the VBMicrolensing 5.5 C++ trajectory math. The
extended-source point-lens magnification (`ESPLMag2`) is a pure JAX port of
VBMicrolensing's precomputed-table lookup (`ESPL.tbl`), so it keeps the C++
engine's speed while being differentiable in `u` and `rho`.

Not yet supported: limb darkening, triple/multiple lenses, binary sources,
astrometry, satellite parallax, caustic/critical-curve geometry.

## HMC knobs

Two attributes not present in VBMicrolensing tune sampler behaviour:

* `vbm.finite_mask` — a fixed per-epoch Boolean array. `True` forces the
  finite-source contour branch, `False` the analytic point-source branch. A
  fixed mask stops `BinaryMag2`'s parameter-dependent point/finite switch from
  putting a small discontinuity in the sampled target. `None` keeps the
  adaptive choice.
* `vbm.grad_tol` — an absolute per-epoch Jacobian-error budget, decoupled from
  `Tol`/`RelTol`, checked with an analytic two-form parabola-disagreement
  estimator (no finite differencing). An epoch that cannot be certified within
  it gets a `NaN` Jacobian row — an honest refusal rather than a silently loose
  derivative. `None` selects a value-tied default.
* `vbm.espl_table_only` — force `ESPLMag2`'s table branch everywhere instead of
  its parameter-dependent switch to the closed-form quadrupole expansion for
  small `rho`. Removes that small primal jump from an HMC target, at the cost of
  the closed form's accuracy for very small `rho` (where the table saturates).
  The table interpolation is bilinear, matching VBMicrolensing exactly; its
  `d/drho` derivative carries a sub-percent bias right at a table-cell edge,
  which averages out over a light curve's epochs.

## Installation

Requires Python ≥ 3.10, a C++17 compiler, and CMake ≥ 3.18.

```bash
pip install -e '.[vbm]'
autodiff-vbm-build-native
```

The build reads the C++ sources shipped with the separately installed
`VBMicrolensing==5.5` and compiles them together with `native/vbm_ffi.cpp`
into a private extension `autodiff_vbm._autodiff_vbm_native`. It does **not**
modify the installed VBMicrolensing package. Rebuild after changing Python,
JAX, the compiler, or VBMicrolensing.

The version is pinned to 5.5 because the integration uses VBMicrolensing's
internal contour data structures, not only its stable Python API.

CPU and float64 only. `jax.jit`, first-order `jax.jvp`, and first-order
`jax.grad` are supported.

## Licensing

autodiff-vbm's own code is MIT (see `LICENSE`). VBMicrolensing is LGPL-3.0 and
is **not** bundled — it is compiled on your machine from the separately
installed package. Distributing a pre-built binary that embeds the native
extension triggers the LGPL-3.0 relinking obligation on the VBMicrolensing
portion; see `NOTICE` for the details. Do not publish a fork or release that
redistributes VBMicrolensing code without first consulting Valerio Bozza.
