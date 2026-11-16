# Scientific conventions and limitations

## Data and likelihood

Input times and t0 use days after JD/HJD 2450000. Supply a consistent time standard; the pipeline does not perform barycentric/time-scale conversions. Photometry is flux, not magnitude. Each dataset has its own source and blend flux, eliminated through weighted linear least squares. The objective is the sum of independent Gaussian flux residual squares. Correlated systematics, error renormalization, limb darkening, orbital motion, xallarap and satellite parallax are not added by this workflow.

All models use the same valid rows and the same time cut, centered on the initial model rather than on a fitted model. Nonfinite rows and nonpositive uncertainties are excluded. With no time window, all valid observations are retained. Photometry paths are resolved relative to the list; the original parent-relative convention is supported only if unambiguous.

## Parameters

Physical parameters are t0, tE, signed u0, q, s, alpha and rho, with positive scales often optimized in logarithmic coordinates. Alpha is in **radians** in the VBMicrolensing center-of-mass convention. These are not the degree-valued alpha coordinates used by some other packages.

Parallax parameters are named **pai1, pai2**, the native VBMicrolensing basis used by the vendored backend. Do not label them pi_E,N and pi_E,E or compare them to an equatorial result without an explicit convention conversion. RA/Dec, the reference epoch recorded in best-fit files and the backend convention must be retained when reproducing a curve. The pipeline does not implement a general cross-library parallax-coordinate conversion.

The legacy `#BEST` first seven values are t0,tE,u0,q,s,alpha,rho. Value 8 (one-based, excluding #BEST) is reserved. Values 9 and 10 are pai1,pai2. Other columns carry legacy fit metadata. Use the supplied readers and examples rather than assuming nine contiguous physical fields.

## Optimization and sampling

The workflow compares PSPL, PSPL with annual parallax, FSPL and FSPL with annual parallax; binary seeds come from an external grid. Binary multistart optimization refines candidates. The continuation gate uses the configured difference in reduced chi-squared between FSPL and static binary, default 1. This is an operational gate, not a calibrated detection significance or Bayesian evidence.

The current binary workflow samples only the top-ranked refined static mode and its parallax branch. Lower-ranked refinement candidates remain available, but are not all automatically sampled or weighted. Reciprocal-s seeds are disabled in the wrapper because the external grid supplies mode seeds; completeness still depends on that grid and refinement. No global mode-recovery guarantee is provided.

NUTS uses local bounded uniform priors in the workers' transformed coordinates, centered around the fitted solution. Their numeric lower/upper bounds and raw coordinate names are saved in the NPZ. Uniformity in a logarithmic coordinate is not uniformity in the physical coordinate. Fluxes are profiled, not integrated over a stated flux prior, so this is a local profiled-flux posterior. It is not automatically a marginal posterior under a population prior.

Raw samples retain chain and draw axes. Sidecar JSON files record effective sample sizes and R-hat diagnostics when computable. Check divergences, trace plots, rank diagnostics, priors, boundaries and alternative modes before interpreting uncertainties. The workflow does not keep extending chains until convergence. The maximum-likelihood draw written as a best fit is distinct from a posterior median. Model selection currently uses chi-squared, not marginal likelihood/evidence.

## Backend

The differentiable backend contains native contour calculations and finite-difference derivative machinery; automatic differentiation does not make finite-source or caustic-crossing numerical errors disappear. Validate tolerances for the event and compare selected curves with an independent physical implementation. The pinned VBMicrolensing/JAX combination has been tested on synthetic inputs, including a short annual-parallax binary sampling check. This does not validate the entire physical parameter space.
