# Migration from the imported tools

The original directory is preserved separately as `microevent-tools-import` in the local workspace. This repository contains a curated implementation, not an in-place rewrite of that archive.

## Main mapping

- `go_long_sleep.py`: replaced by config, pipeline, stage builders and CLI.
- Root JAX/VBM optimization, NUTS and plot scripts: scientific kernels under `src/microevent_tools/kernels/`.
- `native_vbm_emcee/fit_model.py`: `microevent native-fit`.
- Server watch/dispatch scripts: replaced by an explicit per-event `microevent submit` entry point. Automatic filesystem watchers and site-specific batch probes are not distributed.
- autodiff-vbm: source-only snapshot in `vendor/`, retaining its original license and attribution. No upstream commit ID was available in the imported snapshot.
- External autogrid: adapter only; users supply their own grid code/maps/results.

Individual imported-source SHA256 values are recorded in `source-provenance.json`. Those hashes identify the unmodified source inputs, not the refactored files.

## Corrections and behavior changes

- Centralized dataset path resolution and time selection so optimization, sampling and reports use the same observations.
- Removed fixed private server paths and redundant site-specific module searches.
- Added strict TOML validation, per-version locks, stage fingerprints, output checksums, explicit completeness checks and resumable completed stages.
- Corrected the binary NUTS output angle handling: removed a second `pi-alpha` transformation after the backend had already supplied the VBMicrolensing convention.
- Corrected binary parallax best-fit serialization to preserve the reserved column before pai1/pai2, and refreshed fit metadata after sampling.
- Renamed misleading parallax labels to native `pai1/pai2`; no unsupported equatorial conversion is claimed.
- Preserved chain axes, raw coordinate names/prior bounds and sampler diagnostics; chains execute sequentially to respect CPU allocations.
- Model selection reads the MCMC-best static and parallax files consistently.
- Corrected caustic plot trajectories to use radian-valued alpha and the fit backend coordinate convention; annual-parallax diagrams now use the curved trajectory.
- Light-curve plots use unique filenames per model; the stage tracker checks the actual generated files.
- Standalone native photometry now uses the same explicit time convention/path resolution as the pipeline, and accepts the RADec example format.
- Standalone native emcee initial states use an explicit random seed.

## Excluded material

Observed photometry, fit outputs, logs, compiled shared libraries, caches, virtual environments, personal configuration, unrelated GULLS/ML projects and historical server experiments are excluded. Synthetic demo photometry is included. This is not a validation of every historical research result; changed scientific conventions should be checked against saved physical curves before migrating an active analysis.

The primary project has no selected redistribution license yet. Third-party MIT/GPL/LGPL notices are retained in the vendor tree. Resolve licensing of the primary code and any redistribution obligations before making a public release.

## lcbinint backend migration (2026-11-16)

The default analysis now imports lcbinint directly. The old autodiff-vbm facade and vendored source snapshot were removed from the working tree; the previous commits preserve them. Single-lens physics is in point_backend.py, with lcbinint supplying annual trajectories. Normal fits, NUTS, reporting and caustic geometry do not import VBM. `native-fit` is an explicitly optional legacy comparison command.

Parallax outputs change to north/east piEN/piEE with a mandatory NE header. Old VBM pai1/pai2 results are not numerically interchangeable; use a fresh version and refit. Existing output files are not rewritten. Legacy worker filenames and the `.VBM` filename segment remain for command/path compatibility; engine headers identify the actual new implementation.
