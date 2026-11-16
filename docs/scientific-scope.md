# Scientific conventions and limitations

## Data and likelihood

Input times and t0 use days after JD/HJD 2450000. Supply a consistent time standard; the pipeline does not perform barycentric/time-scale conversions. Photometry is flux, not magnitude. Each dataset has its own source and blend flux, eliminated through weighted linear least squares. The objective is the sum of independent Gaussian flux residual squares. Correlated systematics, error renormalization, limb darkening, orbital motion, xallarap and satellite parallax are not added by this workflow.

All models use the same valid rows and the same time cut, centered on the initial model rather than on a fitted model. Nonfinite rows and nonpositive uncertainties are excluded. With no time window, all valid observations are retained. Photometry paths are resolved relative to the list; the original parent-relative convention is supported only if unambiguous.

## Parameters

Physical parameters are t0, tE, signed u0, q, s, alpha and rho, with positive scales often optimized in logarithmic coordinates. Alpha is in **radians** in the VBMicrolensing center-of-mass convention. These are not the degree-valued alpha coordinates used by some other packages.

Parallax parameters are **piEN, piEE**, the north/east components accepted by lcbinint. Keep sky coordinates and a fixed reference epoch. New parallax best-fit files declare `# parallax_basis= NE`. The reader rejects old unlabelled nonzero parallax inputs rather than interpreting VBM pai1/pai2 values as north/east. Start a fresh fit/version; no VBM-basis conversion is implemented.

The legacy `#BEST` first seven values are t0,tE,u0,q,s,alpha,rho. Value 8 (one-based, excluding #BEST) is reserved. Values 9 and 10 are now piEN,piEE **only when the NE header is present**. Old files in Git history used a different basis. Other columns carry legacy fit metadata.

## Optimization and sampling

The workflow compares PSPL, PSPL with annual parallax, FSPL and FSPL with annual parallax; binary seeds come from an external grid. Binary multistart optimization refines candidates. The continuation gate uses the configured difference in reduced chi-squared between FSPL and static binary, default 1. This is an operational gate, not a calibrated detection significance or Bayesian evidence.

The current binary workflow samples only the top-ranked refined static mode and its parallax branch. Lower-ranked refinement candidates remain available, but are not all automatically sampled or weighted. Reciprocal-s seeds are disabled in the wrapper because the external grid supplies mode seeds; completeness still depends on that grid and refinement. No global mode-recovery guarantee is provided.

NUTS uses local bounded uniform priors in the workers' transformed coordinates, centered around the fitted solution. Their numeric lower/upper bounds and raw coordinate names are saved in the NPZ. Uniformity in a logarithmic coordinate is not uniformity in the physical coordinate. Fluxes are profiled, not integrated over a stated flux prior, so this is a local profiled-flux posterior. It is not automatically a marginal posterior under a population prior.

Raw samples retain chain and draw axes. Sidecar JSON files record effective sample sizes and R-hat diagnostics when computable. Check divergences, trace plots, rank diagnostics, priors, boundaries and alternative modes before interpreting uncertainties. The workflow does not keep extending chains until convergence. The maximum-likelihood draw written as a best fit is distinct from a posterior median. Model selection currently uses chi-squared, not marginal likelihood/evidence.

## Backend

The production binary backend is lcbinint's differentiable CPU inverse-ray implementation, pinned to a source commit in pyproject.toml. Finite-source values and derivatives depend on numerical resolution and caustic geometry. Static fits require successful warmup; dynamic annual trajectories use lcbinint's dynamic dispatch. NUTS smoothness, convergence and resolution must be checked for the event. No unconditional speed or accuracy improvement over VBM is claimed.

PSPL uses the exact analytic formula. Uniform FSPL integrates analytic point-lens image contours with 2048/4096 boundary nodes and checks the difference against absolute/relative tolerance. Far from the source (rho/u < 0.001), it uses the PSPL asymptotic limit; this introduces an approximation below the configured default tolerance in the validated domain. This is not a table lookup or a lcbinint single-lens implementation. Nonconverged contour evaluations return NaN instead of silently accepting a coarse result. A node-doubling check is an estimate rather than a rigorous error bound, especially at limb crossings. Validate event-specific accuracy before production inference.

The optional `native-fit` tool remains a separate VBM/emcee comparison path, with VBM pai1/pai2 conventions. Its output must not be used as an NE-labelled input without conversion.
