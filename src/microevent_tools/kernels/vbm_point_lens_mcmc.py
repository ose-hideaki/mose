#!/usr/bin/env python3
"""Run profiled-flux NUTS for an optimized JAX PSPL or FSPL model."""

from __future__ import annotations

# Support both installed modules and the internal subprocess entry points.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "microevent_tools.kernels"


import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value

from .point_backend import PointLens
from .jax_binary_static_fit import profile_flux, read_best, read_data
from ..sampling import nuts_diagnostics

jax.config.update("jax_enable_x64", True)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--list", required=True, type=Path, dest="list_path")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bestfit-output", type=Path)
    parser.add_argument("--finite-source", action="store_true")
    parser.add_argument("--parallax", action="store_true")
    parser.add_argument("--radec", type=Path)
    parser.add_argument("--time-window", type=float, default=300.0)
    parser.add_argument("--data-center", type=float,
                        help="fixed center used only for selecting data")
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tree-depth", type=int, default=10)
    parser.add_argument("--tol", type=float, default=1e-4)
    return parser.parse_args()


def read_parallax(path):
    from .jax_binary_parallax_fit import read_parallax as reader
    return reader(path)


def read_reference_time(path: Path, default: float) -> float:
    for line in path.read_text().splitlines():
        if line.startswith("# t_ref="):
            return float(line.split()[2])
    return default


def main():
    args = arguments()
    defaults = {"t0": 0.0, "tE": 10.0, "u0": 0.1, "q": 0.0,
                "s": 0.0, "alpha": 0.0, "rho": 0.002}
    initial = read_best(args.model, defaults)
    data_center = initial["t0"] if args.data_center is None else args.data_center
    datasets, times = read_data(args.list_path, data_center, args.time_window)
    times = jnp.asarray(times)
    vbm = PointLens()
    vbm.Tol = args.tol
    vbm.RelTol = args.tol

    reference_time = initial["t0"]
    if args.parallax:
        from .jax_binary_parallax_fit import read_radec

        radec = args.radec or args.list_path.parent.parent / "RADec"
        if not radec.is_file():
            raise FileNotFoundError(radec)
        ra, dec = read_radec(radec)
        vbm.SetObjectCoordinates(f"{ra} {dec}")
        reference_time = read_reference_time(args.model, initial["t0"])
        vbm.t0_par_fixed = 1
        vbm.t0_par = reference_time
        initial["piEN"], initial["piEE"] = read_parallax(args.model)

    center_values = [initial["u0"] if args.parallax else np.log(abs(initial["u0"])),
                     np.log(initial["tE"]), initial["t0"]]
    widths = [max(0.25, 2.0 * abs(initial["u0"])) if args.parallax else 1.5,
              0.8, 1.0]
    names = ["u0", "tE", "t0"]
    if args.finite_source:
        center_values.append(np.log(initial["rho"]))
        widths.append(np.log(20.0))
        names.append("rho")
    if args.parallax:
        center_values.extend([initial["piEN"], initial["piEE"]])
        widths.extend([0.5, 0.5])
        names.extend(["piEN", "piEE"])
    center = np.asarray(center_values)
    widths = np.asarray(widths)

    def objective(theta):
        if args.parallax and args.finite_source:
            magnification = vbm.ESPLLightCurveParallax(theta, times)[0]
        elif args.parallax:
            magnification = vbm.PSPLLightCurveParallax(theta, times)[0]
        elif args.finite_source:
            magnification = vbm.ESPLLightCurve(theta, times)[0]
        else:
            magnification = vbm.PSPLLightCurve(theta, times)[0]
        return sum(profile_flux(
            magnification[item["slice"]], item["flux"], item["error"]
        )[0] for item in datasets)

    def model():
        theta = jnp.stack([
            numpyro.sample(
                f"raw_{index}",
                dist.Uniform(center[index] - widths[index], center[index] + widths[index]),
            )
            for index in range(len(center))
        ])
        numpyro.factor("profile_log_likelihood", -0.5 * objective(theta))

    init = {f"raw_{index}": center[index] for index in range(len(center))}
    sampler = MCMC(
        NUTS(model, init_strategy=init_to_value(values=init),
             max_tree_depth=args.max_tree_depth),
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.chains,
        chain_method="sequential",
    )
    sampler.run(
        jax.random.PRNGKey(args.seed),
        extra_fields=("num_steps", "diverging", "potential_energy"),
    )
    samples = sampler.get_samples(group_by_chain=False)
    raw = np.column_stack([
        np.asarray(samples[f"raw_{index}"]) for index in range(len(center))
    ])
    physical = np.column_stack([
        raw[:, 0] if args.parallax else np.exp(raw[:, 0]),
        np.exp(raw[:, 1]), raw[:, 2]
    ])
    if args.finite_source:
        physical = np.column_stack([physical, np.exp(raw[:, 3])])
    if args.parallax:
        physical = np.column_stack([physical, raw[:, -2], raw[:, -1]])
    extra = sampler.get_extra_fields(group_by_chain=False)
    grouped_raw, diagnostics = nuts_diagnostics(sampler)
    energy = np.asarray(extra["potential_energy"]).reshape(-1)
    best_index = int(np.argmin(energy))
    best_chi2 = float(objective(jnp.asarray(raw[best_index])))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output, samples=physical, raw_samples=raw,
        num_steps=np.asarray(extra["num_steps"]),
        diverging=np.asarray(extra["diverging"]), potential_energy=energy,
        raw_samples_by_chain=grouped_raw, parameters=np.asarray(names),
    )
    quantiles = np.percentile(physical, [16, 50, 84], axis=0)
    summary = {
        "model": ("FSPL" if args.finite_source else "PSPL")
                 + ("_PARALLAX" if args.parallax else ""),
        "parameters": names,
        "q16": quantiles[0].tolist(), "median": quantiles[1].tolist(),
        "q84": quantiles[2].tolist(), "warmup": args.warmup,
        "samples": args.samples, "chains": args.chains,
        "divergences": int(np.sum(extra["diverging"])),
        "max_steps": int(np.max(extra["num_steps"])),
        "best_sample_index": best_index, "best_sample_chi2": best_chi2,
        "diagnostics": diagnostics,
        "raw_prior_lower": (center-widths).tolist(), "raw_prior_upper": (center+widths).tolist(),
    }
    Path(f"{args.output}.json").write_text(json.dumps(summary, indent=2) + "\n")

    if args.bestfit_output:
        lines = args.model.read_text().splitlines()
        ndata = int(times.shape[0])
        nparam = len(center) + 2 * len(datasets)
        dof = ndata - nparam
        # Legacy ordering is t0, tE, u0, ..., rho.
        replacements = [physical[best_index, 2], physical[best_index, 1],
                        physical[best_index, 0]]
        if args.finite_source:
            replacements.append(physical[best_index, 3])
        for index, line in enumerate(lines):
            if line.startswith("#chi2/dof="):
                fields = line.split()
                fields[1] = f"{best_chi2:.12e}"
                for name, value in (
                    ("Ndata=", ndata), ("nparam=", nparam),
                    ("redchi2=", best_chi2 / dof),
                ):
                    field_index = fields.index(name)
                    fields[field_index + 1] = (
                        f"{value:.12e}" if name == "redchi2=" else str(value)
                    )
                lines[index] = " ".join(fields)
            elif line.startswith("#BEST"):
                fields = line.split()
                fields[1] = f"{replacements[0]:.12e}"
                fields[2] = f"{replacements[1]:.12e}"
                fields[3] = f"{replacements[2]:.12e}"
                if args.finite_source:
                    fields[7] = f"{replacements[3]:.12e}"
                if args.parallax:
                    pi_offset = 4 if args.finite_source else 3
                    fields[9] = f"{physical[best_index, pi_offset]:.12e}"
                    fields[10] = f"{physical[best_index, pi_offset + 1]:.12e}"
                lines[index] = " ".join(fields)
        if args.parallax and not any(line.startswith("# t_ref=") for line in lines):
            lines.append(f"# t_ref= {reference_time:.12e}")
        lines += [f"# MCMC_BEST_SAMPLE_INDEX= {best_index}",
                  f"# MCMC_BEST_SAMPLE_CHI2= {best_chi2:.12e}"]
        args.bestfit_output.write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
