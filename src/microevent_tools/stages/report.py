"""Build reproducible model-comparison tables and physical light-curve plots."""
from ..execution import Step, kernel
from ..paths import suffix
from .point import MODELS


def build(config, paths, center, window, *, point_only, post=False):
    model_args = []
    options = ("--pspl", "--pspl-parallax", "--fspl", "--fspl-parallax")
    best = []
    for option, (name, _) in zip(options, MODELS):
        value = suffix(paths.point(name, post), ".MCMC.bestfit")
        model_args.extend((option, value)); best.append(value)
    condition = "always" if config.stage == "point" else ("point_only" if point_only else "binary")
    mode = "point" if point_only else "binary"
    extra = ("--point-only",) if point_only else ()
    table = Step(f"report_{mode}", kernel(config, "summarize_bestfits", *model_args,
        "--binary-prefix", paths.binary, "--parallax-suffix", ".PARALLAX.VBM.MCMC.bestfit",
        "--static-suffix", ".MCMC.bestfit",
        "--output", paths.summary, *extra), (paths.summary,), tuple(best), condition)
    plot = paths.output / "result" / f"{paths.model}.{mode.upper()}_LIGHTCURVES.png"
    binary_args = () if point_only else ("--static", suffix(paths.binary, ".rank1.MCMC.bestfit"),
        "--parallax", suffix(paths.binary, ".rank1.PARALLAX.VBM.MCMC.bestfit"))
    slugs = ["pspl", "fspl", "pspl_parallax", "fspl_parallax"]
    if not point_only:
        slugs += ["static_binary", "parallax_binary"]
    images = tuple(plot.with_name(f"{plot.stem}.{slug}{plot.suffix}") for slug in slugs)
    image = Step(f"plot_{mode}", kernel(config, "plot_bestfit_lightcurves", *model_args,
        "--list", paths.data_list, "--radec", paths.radec, "--output", plot,
        "--window", window, "--time-window", window, "--data-center", center,
        *binary_args, *extra), images, (*best, *[x for x in binary_args if hasattr(x, 'is_file')]), condition)
    return [table, image]
