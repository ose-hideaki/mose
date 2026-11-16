#!/usr/bin/env python3
"""Run local NUTS around one optimized static or annual-parallax binary mode."""

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

import os
from . import autodiff_vbm_backend as lcbinint
from .jax_binary_static_fit import make_objective as make_static_objective
from .jax_binary_static_fit import pack as pack_static
from .jax_binary_static_fit import read_best, read_data
from .jax_binary_static_fit import unpack as unpack_static
from ..sampling import nuts_diagnostics

jax.config.update("jax_enable_x64", True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--list", required=True, type=Path, dest="list_path")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bestfit-output", type=Path,
                        help="Write the maximum-posterior sample as a reusable model file")
    parser.add_argument("--time-window", type=float, default=300.0)
    parser.add_argument("--data-center", type=float,
                        help="fixed center used only for selecting data")
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tree-depth", type=int, default=10)
    parser.add_argument("--parallax", action="store_true")
    parser.add_argument("--radec", type=Path)
    return parser.parse_args()


def read_parallax(path: Path) -> tuple[float, float]:
    for line in path.read_text().splitlines():
        if line.startswith("#BEST"):
            values = [float(value) for value in line.split()[1:]]
            if len(values) >= 10:
                return values[8], values[9]
    raise ValueError(f"No pai1/pai2 fields in {path}")


def read_reference_time(path: Path, default: float) -> float:
    for line in path.read_text().splitlines():
        if line.startswith("# t_ref="):
            return float(line.split()[2])
    return default


def main() -> None:
    args = parse_args()
    defaults = {"t0": 0.0, "tE": 1.0, "u0": 0.1, "q": 1e-3, "s": 1.0, "alpha": 0.0, "rho": 2e-3}
    initial = read_best(args.model, defaults)
    data_center = initial["t0"] if args.data_center is None else args.data_center
    datasets, times = read_data(args.list_path, data_center, args.time_window)
    if args.parallax:
        from .jax_binary_parallax_fit import (
            make_curve, make_objective, pack, read_radec, to_internal, unpack,
        )

        radec = args.radec or args.list_path.parent.parent / "RADec"
        ra, dec = read_radec(radec)
        initial["pai1"], initial["pai2"] = read_parallax(args.model)
        initial = to_internal(initial)
        reference_time = read_reference_time(args.model, initial["t0"])
        sky = lcbinint.obs.SkyCoord(ra, dec)
        curve = make_curve(sky, reference_time)
        objective = make_objective(curve, times, datasets)
        center = pack(initial)
        unpack_parameters = unpack
        parameter_names = ["t0", "tE", "u0", "q", "s", "alpha", "rho", "pai1", "pai2"]
        widths = np.asarray([0.03, 0.08, 0.01, 0.10, 0.10, 0.10, np.log(1.6), 0.25, 0.25])
    else:
        curve = lcbinint.LightCurve(options=lcbinint.Options(jax=True, coordinates="vbm_center_of_mass", tol=1e-4, reltol=1e-4, max_source_bins=400))
        report = curve.warmup(times, initial)
        if not report.all_calibrated:
            raise RuntimeError("lcbinint JAX warmup did not calibrate every epoch")
        objective = make_static_objective(curve, times, datasets)
        center = pack_static(initial)
        unpack_parameters = unpack_static
        parameter_names = ["t0", "tE", "u0", "q", "s", "alpha", "rho"]
        widths = np.asarray([0.03, 0.08, 0.01, 0.10, 0.10, 0.10, np.log(1.6)])

    dimension = len(center)
    lower = center - widths
    upper = center + widths
    lower[2] = max(lower[2], -5.0)
    upper[2] = min(upper[2], 5.0)
    if np.any(lower >= upper):
        raise ValueError("MCMC initial model is outside the allowed parameter bounds")

    def model():
        raw = jnp.stack([
            numpyro.sample(f"raw_{index}", dist.Uniform(lower[index], upper[index]))
            for index in range(dimension)
        ])
        numpyro.factor("profile_log_likelihood", -0.5 * objective(raw))

    init = {f"raw_{index}": center[index] for index in range(dimension)}
    sampler = MCMC(
        NUTS(model, init_strategy=init_to_value(values=init), max_tree_depth=args.max_tree_depth),
        num_warmup=args.warmup, num_samples=args.samples, num_chains=args.chains,
        chain_method="sequential",
    )
    sampler.run(
        jax.random.PRNGKey(args.seed),
        extra_fields=("num_steps", "diverging", "potential_energy"),
    )
    raw_samples = np.column_stack([np.asarray(sampler.get_samples()[f"raw_{index}"]) for index in range(dimension)])
    physical = np.asarray([[float(value) for value in unpack_parameters(jnp.asarray(row)).values()] for row in raw_samples])
    # The autodiff-vbm facade already uses the output VBM angle convention.
    # Applying the old lcbinint pi-alpha conversion would change the model.
    grouped_raw, diagnostics = nuts_diagnostics(sampler)
    extra = sampler.get_extra_fields()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    potential_energy = np.asarray(extra["potential_energy"]).reshape(-1)
    np.savez_compressed(
        args.output, samples=physical, raw_samples=raw_samples,
        num_steps=np.asarray(extra["num_steps"]),
        diverging=np.asarray(extra["diverging"]),
        potential_energy=potential_energy,
        raw_samples_by_chain=grouped_raw, parameters=np.asarray(parameter_names),
    )
    best_index = int(np.argmin(potential_energy))
    best_chi2 = float(objective(jnp.asarray(raw_samples[best_index])))
    if args.bestfit_output is not None:
        lines = args.model.read_text().splitlines()
        for index, line in enumerate(lines):
            if line.startswith("#chi2/dof="):
                fields = line.split()
                fields[1] = f"{best_chi2:.12e}"
                ndata = len(times); nparam = dimension + 2 * len(datasets)
                for name, value in (("Ndata=", ndata), ("nparam=", nparam),
                                    ("redchi2=", best_chi2 / max(1, ndata-nparam))):
                    if name in fields:
                        fields[fields.index(name)+1] = str(value)
                lines[index] = " ".join(fields)
            elif line.startswith("#BEST"):
                fields = line.split()
                # Legacy column 8 (values[7]) is reserved; parallax lives at
                # columns 9 and 10, as in the optimizer and point-lens workers.
                offsets = list(range(1, 8)) + ([9, 10] if args.parallax else [])
                while len(fields) <= max(offsets):
                    fields.append("0")
                for offset, value in zip(offsets, physical[best_index]):
                    fields[offset] = f"{value:.12e}"
                lines[index] = " ".join(fields)
        lines.append(f"# MCMC_BEST_SAMPLE_INDEX= {best_index}")
        lines.append(f"# MCMC_BEST_SAMPLE_CHI2= {best_chi2:.12e}")
        args.bestfit_output.parent.mkdir(parents=True, exist_ok=True)
        args.bestfit_output.write_text("\n".join(lines) + "\n")
    quantiles = np.percentile(physical, [16, 50, 84], axis=0)
    summary = {
        "parameters": parameter_names,
        "parallax": args.parallax,
        "q16": quantiles[0].tolist(), "median": quantiles[1].tolist(), "q84": quantiles[2].tolist(),
        "warmup": args.warmup, "samples": args.samples, "chains": args.chains,
        "divergences": int(np.sum(extra["diverging"])), "max_steps": int(np.max(extra["num_steps"])),
        "best_sample_index": best_index, "best_sample_chi2": best_chi2,
        "diagnostics": diagnostics, "raw_prior_lower": lower.tolist(), "raw_prior_upper": upper.tolist(),
    }
    Path(f"{args.output}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
