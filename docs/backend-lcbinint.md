# lcbinint CPU backend

Upstream: https://github.com/NunotaKansuke/lcbinint

Pinned commit: `b51c00c566667fc2ed1fe640543219f480642002`.

The public `lcbinint.LightCurve` with `Options(jax=True, coordinates="vbm_center_of_mass")` evaluates binary magnification and gradients. Annual models use `Model(parallax=True, sky=..., t_ref=...)` and **piEN/piEE**, not native VBM pai1/pai2. We retain radian alpha and the same center-of-mass static trajectory. Obtain caustics and trajectories from the same curve coordinate convention.

lcbinint currently exposes binary/triple lenses. PSPL is analytic JAX and uniform FSPL is this package's analytic-image contour integration, independently checked against a physical reference. Annual single-lens paths come from lcbinint source_trajectory; dummy binary geometry is supplied only to that trajectory helper, never used to calculate single-lens magnification.

Use JAX/JAXlib 0.6.2 and NumPyro 0.19.0 in a separate environment. lcbinint's native build imports `jax.ffi`, which the old mose JAX 0.4.35 environment does not expose at that location. Build without isolation after installing the pinned JAX so the FFI headers and runtime match. Native dependencies include GSL and Boost headers. No binary or upstream source is vendored in mose.

Start with `microevent plan examples/point.toml`, then a fresh event version. Never resume old VBM-version outputs under this backend. `native-fit` is an optional comparison utility and is not part of the default pipeline.

Trajectory helpers use the JAX path on CPU, including plots. During migration, the upstream native annual trajectory helper returned NaNs on the Linux CI runner while JAX returned finite values. We therefore avoid the native annual helper in production and check plotted annual trajectories for finite values in regression tests. This is a workaround, not a claim to have corrected upstream native ephemeris behavior.
