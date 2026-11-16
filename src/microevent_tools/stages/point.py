"""Build the four point-lens optimization/sampling branches."""
from ..execution import Step, kernel
from ..paths import suffix


MODELS = (("PSPL", ()), ("PSPL.PARALLAX", ("--parallax",)),
          ("FSPL", ("--finite-source",)), ("FSPL.PARALLAX", ("--finite-source", "--parallax")))


def build(config, paths, center, window, *, post=False):
    steps = []
    for i, (name, flags) in enumerate(MODELS):
        fit = paths.point(name, post)
        if post:
            initial = paths.final
        elif name == "PSPL":
            initial = paths.initial
        elif name in ("PSPL.PARALLAX", "FSPL"):
            initial = suffix(paths.point("PSPL"), ".MCMC.bestfit")
        else:
            initial = suffix(paths.point("PSPL.PARALLAX"), ".MCMC.bestfit")
        data = ("--list", paths.data_list, "--time-window", window, "--data-center", center)
        sky = ("--radec", paths.radec) if "--parallax" in flags else ()
        condition = "binary" if post else "always"
        stem = ("post_" if post else "") + name.lower().replace(".", "_")
        steps.append(Step(stem+"_fit", kernel(config, "vbm_point_lens_fit",
            "--model", initial, *data, "--output", fit, *flags, *sky), (fit,), (initial,), condition))
        chain = suffix(fit, ".mcmc.npz"); best = suffix(fit, ".MCMC.bestfit")
        steps.append(Step(stem+"_mcmc", kernel(config, "vbm_point_lens_mcmc",
            "--model", fit, *data, "--output", chain, "--bestfit-output", best,
            "--warmup", config.warmup, "--samples", config.samples, "--chains", config.chains,
            "--seed", config.seed + i*50 + (200 if post else 0), *flags, *sky),
            (chain, best, suffix(chain, ".json")), (fit,), condition))
    return steps
