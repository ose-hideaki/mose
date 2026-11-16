#!/usr/bin/env python3
"""Fit PSPL or uniform-source FSPL with autodiff-vbm."""

from __future__ import annotations

# Support both installed modules and the internal subprocess entry points.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "microevent_tools.kernels"


import argparse
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

from autodiff_vbm import AutodiffVBM
from .jax_binary_static_fit import legacy_best, profile_flux, read_best, read_data

jax.config.update("jax_enable_x64", True)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--list", required=True, type=Path, dest="list_path")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--finite-source", action="store_true")
    parser.add_argument("--parallax", action="store_true")
    parser.add_argument("--radec", type=Path)
    parser.add_argument("--rho-init", type=float, default=0.002)
    parser.add_argument("--time-window", type=float, default=300.0)
    parser.add_argument("--data-center", type=float,
                        help="fixed center used only for selecting data")
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument(
        "--maxiter", type=int,
        help="optional optimizer iteration limit (primarily useful for smoke tests)",
    )
    return parser.parse_args()


def read_parallax(path: Path) -> tuple[float, float]:
    for line in path.read_text().splitlines():
        if line.startswith("#BEST"):
            values = [float(value) for value in line.split()[1:]]
            if len(values) >= 10:
                return values[8], values[9]
    return 0.0, 0.0


def read_reference_time(path: Path, default: float) -> float:
    for line in path.read_text().splitlines():
        if line.startswith("# t_ref="):
            return float(line.split()[2])
    return default


def read_radec(path: Path) -> tuple[str, str]:
    match = re.search(r"\bRA=\s*(\S+)\s+Dec=\s*(\S+)", path.read_text())
    if match is None:
        raise ValueError(f"Cannot read RA/Dec from {path}")
    return match.group(1), match.group(2)


def finite_source_starts(initial_rho: float) -> tuple[float, ...]:
    """Return rho starts with the input model's value tried first."""
    candidates = (initial_rho, 2e-4, 2e-3, 1e-2, 5e-2, 1e-1)
    starts = []
    for rho in candidates:
        if not np.isfinite(rho) or not 1e-6 <= rho <= 2.0:
            continue
        if not any(np.isclose(rho, value, rtol=0.0, atol=1e-15) for value in starts):
            starts.append(float(rho))
    return tuple(starts)


def main():
    args = arguments()
    defaults = {"t0": 0.0, "tE": 10.0, "u0": 0.1, "q": 1e-3,
                "s": 1.0, "alpha": 0.0, "rho": args.rho_init}
    initial = read_best(args.model, defaults)
    data_center = initial["t0"] if args.data_center is None else args.data_center
    datasets, times = read_data(args.list_path, data_center, args.time_window)
    times_jax = jnp.asarray(times)
    vbm = AutodiffVBM()
    vbm.Tol = args.tol
    vbm.RelTol = args.tol
    vbm.espl_table_only = args.finite_source
    reference_time = initial["t0"]
    if args.parallax:
        radec = args.radec or args.list_path.parent.parent / "RADec"
        if not radec.is_file():
            raise FileNotFoundError(radec)
        ra, dec = read_radec(radec)
        vbm.SetObjectCoordinates(f"{ra} {dec}")
        reference_time = read_reference_time(args.model, initial["t0"])
        vbm.t0_par_fixed = 1
        vbm.t0_par = reference_time
        initial["pai1"], initial["pai2"] = read_parallax(args.model)

    if args.parallax:
        center_values = [initial["u0"], np.log(initial["tE"]), initial["t0"]]
        if args.finite_source:
            center_values.append(np.log(initial["rho"]))
        center_values.extend([initial["pai1"], initial["pai2"]])
        center = np.asarray(center_values)
        rho_starts = finite_source_starts(initial["rho"]) if args.finite_source else (None,)
        if args.finite_source:
            # Continue from the PSPL-parallax solution while scanning rho.
            pi_starts = ((initial["pai1"], initial["pai2"]),)
        else:
            pi_starts = (
                (initial["pai1"], initial["pai2"]),
                (0.0, 0.0), (0.1, 0.0), (-0.1, 0.0),
                (0.0, 0.1), (0.0, -0.1),
            )
    elif args.finite_source:
        center = np.asarray([np.log(abs(initial["u0"])), np.log(initial["tE"]),
                             initial["t0"], np.log(initial["rho"])])
        rho_starts = finite_source_starts(initial["rho"])
        pi_starts = (None,)
    else:
        center = np.asarray([np.log(abs(initial["u0"])), np.log(initial["tE"]), initial["t0"]])
        rho_starts = (None,)
        pi_starts = (None,)

    def objective(theta):
        if args.parallax and args.finite_source:
            magnification = vbm.ESPLLightCurveParallax(theta, times_jax)[0]
        elif args.parallax:
            magnification = vbm.PSPLLightCurveParallax(theta, times_jax)[0]
        elif args.finite_source:
            magnification = vbm.ESPLLightCurve(theta, times_jax)[0]
        else:
            magnification = vbm.PSPLLightCurve(theta, times_jax)[0]
        total = 0.0
        for dataset in datasets:
            total += profile_flux(magnification[dataset["slice"]], dataset["flux"], dataset["error"])[0]
        return total

    value_gradient = jax.jit(jax.value_and_grad(objective))
    def function(theta):
        value, gradient = value_gradient(jnp.asarray(theta))
        return float(value), np.asarray(gradient, dtype=float)

    fits = []
    for rho in rho_starts:
        for pi_start in pi_starts:
            start = center.copy()
            if rho is not None:
                start[3] = np.log(rho)
            if pi_start is not None:
                start[-2:] = pi_start
            if args.parallax:
                bounds = [(-5.0, 5.0), (center[1] - 2, center[1] + 2),
                          (center[2] - 2, center[2] + 2)]
            else:
                bounds = [(center[0] - 3, center[0] + 3),
                          (center[1] - 2, center[1] + 2),
                          (center[2] - 2, center[2] + 2)]
            if args.finite_source:
                bounds.append((np.log(1e-6), np.log(2.0)))
            if args.parallax:
                bounds.extend([(-2.0, 2.0), (-2.0, 2.0)])
            options = {"maxiter": args.maxiter} if args.maxiter is not None else None
            fits.append(minimize(
                function, start, jac=True, method="L-BFGS-B", bounds=bounds,
                options=options,
            ))
    fit = min(fits, key=lambda item: item.fun if np.isfinite(item.fun) else np.inf)
    if not np.isfinite(fit.fun):
        raise RuntimeError("autodiff-vbm point-lens optimization failed")

    u0 = fit.x[0] if args.parallax else np.exp(fit.x[0])
    tE, t0 = np.exp(fit.x[1]), fit.x[2]
    rho = np.exp(fit.x[3]) if args.finite_source else 0.0
    pai1, pai2 = fit.x[-2:] if args.parallax else (0.0, 0.0)
    parameters = {"t0": t0, "tE": tE, "u0": u0, "q": 0.0, "s": 0.0,
                  "alpha": 0.0, "rho": rho, "pai1": pai1, "pai2": pai2}
    if args.parallax and args.finite_source:
        magnification = vbm.ESPLLightCurveParallax(fit.x, times_jax)[0]
    elif args.parallax:
        magnification = vbm.PSPLLightCurveParallax(fit.x, times_jax)[0]
    elif args.finite_source:
        magnification = vbm.ESPLLightCurve(fit.x, times_jax)[0]
    else:
        magnification = vbm.PSPLLightCurve(fit.x, times_jax)[0]
    results = []
    for dataset in datasets:
        chi2, fs, fb = profile_flux(
            magnification[dataset["slice"]], dataset["flux"], dataset["error"]
        )
        results.append((float(chi2), float(fs), float(fb)))
    ndata = len(times)
    nparam = len(fit.x) + 2 * len(datasets)
    model_name = "FSPL" if args.finite_source else "PSPL"
    if args.parallax:
        model_name += "_PARALLAX"
    legacy = legacy_best(parameters, results)
    legacy[8], legacy[9] = pai1, pai2
    lines = [
        "# generated by vbm_point_lens_fit.py",
        f"# engine= autodiff-vbm model= {model_name}",
    ]
    if args.parallax:
        lines.append(f"# t_ref= {reference_time:.12e}")
    for index, (dataset, result) in enumerate(zip(datasets, results)):
        lines.append(f"#chi2band {index} {result[0]:.12e} {dataset['observatory']} {dataset['band']}")
    lines += [
        f"#chi2/dof= {fit.fun:.12e} Ndata= {ndata} nparam= {nparam} redchi2= {fit.fun/(ndata-nparam):.12e}",
        "#BEST " + " ".join(f"{value:.12e}" for value in legacy),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"{model_name} chi2={fit.fun:.12f} output={args.output}")


if __name__ == "__main__":
    main()
