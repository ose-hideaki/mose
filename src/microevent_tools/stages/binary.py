"""Binary refinement, gate, local NUTS, annual parallax and solution selection."""
from ..execution import Step, kernel
from ..paths import suffix


def build(config, paths, center, window):
    data = ("--list", paths.data_list, "--time-window", window, "--data-center", center)
    rank = suffix(paths.binary, ".rank1")
    static_best = suffix(rank, ".MCMC.bestfit")
    static_chain = suffix(rank, ".mcmc.npz")
    parallax = suffix(rank, ".PARALLAX.VBM")
    parallax_chain = suffix(parallax, ".mcmc.npz")
    # Preserve the optimization filename for the selector while retaining the MCMC best separately.
    parallax_best = suffix(parallax, ".MCMC.bestfit")
    sampling = ("--warmup", config.warmup, "--samples", config.samples, "--chains", config.chains)
    return [
        Step("binary_refine", kernel(config, "jax_binary_multistart", "--model", suffix(paths.modes, ".mode1"),
            "--candidates-file", suffix(paths.modes, ".candidates"), *data, "--output-prefix", paths.binary,
            "--max-seed-modes", config.max_modes, "--max-stages", config.max_stages,
            "--keep", config.keep, "--delta-chi2", config.delta_chi2,
            "--no-reciprocal-seeds", "--seed", config.seed),
            (rank, suffix(paths.binary, ".summary")), (suffix(paths.modes, ".candidates"), suffix(paths.modes, ".mode1"))),
        Step("binary_gate", kernel(config, "check_binary_improvement",
            "--fspl", suffix(paths.point("FSPL"), ".MCMC.bestfit"), "--binary", rank,
            "--threshold", config.binary_threshold, "--output", suffix(paths.binary, ".improvement.tsv"),
            *(("--always-continue",) if config.always_continue else ())),
            (suffix(paths.binary, ".improvement.tsv"),), (rank, suffix(paths.point("FSPL"), ".MCMC.bestfit")), accepted_codes=(0, 20)),
        Step("binary_static_mcmc", kernel(config, "jax_binary_mcmc", "--model", rank, *data,
            "--output", static_chain, "--bestfit-output", static_best, *sampling, "--seed", config.seed),
            (static_chain, static_best, suffix(static_chain, ".json")), (rank,), "binary"),
        Step("binary_parallax_fit", kernel(config, "jax_binary_parallax_fit", "--model", static_best, *data,
            "--radec", paths.radec, "--point-parallax-model", suffix(paths.point("FSPL.PARALLAX"), ".MCMC.bestfit"),
            "--output", parallax, "--max-stages", config.max_stages), (parallax,),
            (static_best, suffix(paths.point("FSPL.PARALLAX"), ".MCMC.bestfit")), "binary"),
        Step("binary_parallax_mcmc", kernel(config, "jax_binary_mcmc", "--model", parallax, *data,
            "--radec", paths.radec, "--parallax", "--output", parallax_chain,
            "--bestfit-output", parallax_best, *sampling, "--seed", config.seed + 1000),
            (parallax_chain, parallax_best, suffix(parallax_chain, ".json")), (parallax,), "binary"),
        Step("select_binary", kernel(config, "select_binary_solution", "--prefix", paths.binary,
            "--max-rank", 1, "--parallax-suffix", ".PARALLAX.VBM.MCMC.bestfit",
            "--static-suffix", ".MCMC.bestfit", "--output", paths.final,
            "--summary", suffix(paths.final, ".summary")), (paths.final, suffix(paths.final, ".summary")),
            (static_best, parallax_best), "binary"),
    ]
