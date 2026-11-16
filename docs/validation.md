# Local validation record

Date: 2026-11-16. Platform: macOS, CPU; Python 3.12, NumPy 1.26.4, JAX/JAXlib 0.4.35, NumPyro 0.15.3, VBMicrolensing 5.5, emcee 3.1.6.

- 12 unittest checks passed: configuration and fixed data cuts, relative paths, SGE resources/script syntax, stage reuse and invalidation, gate handling, native PSPL analytic agreement and flux recovery, finite signed-u0 binary magnifications, deterministic short emcee runs, native RADec/data conventions, JAX magnification/gradient, short binary annual-parallax NUTS serialization.
- Vendored native extension built successfully from source on this Mac.
- Root package wheel built successfully without bundling the native extension.
- Synthetic point workflow ran all four point models through optimization, short local NUTS, summary and four light-curve plots. Warmup/draws were intentionally only 8/8 and one chain; this was an execution check, not an inference-quality result. Repeating the command reused all ten completed stages. The generated light-curve image was visually checked.
- Full binary workflow plan generated successfully. Its physical NUTS kernel is covered by a separate short synthetic parallax test. The external-grid-to-final-binary workflow has **not** been run end to end: an external complete grid/map installation is required.
- GitHub Actions definitions cover the portable/native checks and a separately compiled JAX backend on Linux. This record describes local results; it does not claim a remote CI result before it is available.

No real event analysis or convergence claim is made by this validation.
