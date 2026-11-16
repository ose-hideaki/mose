#!/usr/bin/env python3
"""Fit a static finite-source binary lens with the vendored autodiff-vbm JAX backend."""

from __future__ import annotations

# Support both installed modules and the internal subprocess entry points.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "microevent_tools.kernels"


import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

import os
from . import autodiff_vbm_backend as lcbinint

jax.config.update("jax_enable_x64", True)


PARAMETER_NAMES = ("t0", "tE", "u0", "q", "s", "alpha", "rho")


def read_best(path: Path, defaults: dict[str, float]) -> dict[str, float]:
    for line in path.read_text().splitlines():
        if not line.startswith("#BEST"):
            continue
        values = [float(value) for value in line.split()[1:]]
        if len(values) < 3:
            break
        result = dict(defaults)
        result.update(t0=values[0], tE=values[1], u0=values[2])
        if len(values) >= 7:
            result.update(q=values[3], s=values[4], alpha=values[5], rho=values[6])
        for name in ("tE", "q", "s", "rho"):
            if result[name] <= 0.0:
                result[name] = defaults[name]
        return result
    raise ValueError(f"No usable #BEST in {path}")


def read_data(list_path: Path, center: float, time_window: float):
    from ..data import load_datasets
    datasets = []
    for item in load_datasets(list_path, center, time_window):
        values = item.values
        datasets.append(
            {
                "path": item.path,
                "observatory": item.observatory,
                "band": item.band,
                "time": values[:, 0],
                "flux": jnp.asarray(values[:, 1]),
                "error": jnp.asarray(values[:, 2]),
            }
        )
    times = np.concatenate([dataset["time"] for dataset in datasets])
    offset = 0
    for dataset in datasets:
        size = len(dataset["time"])
        dataset["slice"] = slice(offset, offset + size)
        offset += size
    return datasets, times


def profile_flux(magnification, flux, error):
    weight = 1.0 / jnp.square(error)
    saa = jnp.sum(weight * magnification * magnification)
    sa = jnp.sum(weight * magnification)
    s1 = jnp.sum(weight)
    say = jnp.sum(weight * magnification * flux)
    sy = jnp.sum(weight * flux)
    determinant = saa * s1 - sa * sa
    source_flux = (say * s1 - sy * sa) / determinant
    blend_flux = (sy * saa - say * sa) / determinant
    residual = (flux - source_flux * magnification - blend_flux) / error
    return jnp.sum(residual * residual), source_flux, blend_flux


def unpack(raw):
    return {
        "t0": raw[0],
        "tE": jnp.exp(raw[1]),
        "u0": raw[2],
        "q": jnp.exp(raw[3]),
        "s": jnp.exp(raw[4]),
        "alpha": raw[5],
        "rho": jnp.exp(raw[6]),
    }


def pack(parameters: dict[str, float]):
    return np.asarray(
        [
            parameters["t0"],
            np.log(parameters["tE"]),
            parameters["u0"],
            np.log(parameters["q"]),
            np.log(parameters["s"]),
            parameters["alpha"],
            np.log(parameters["rho"]),
        ]
    )


def physical(raw):
    values = unpack(jnp.asarray(raw))
    return {name: float(values[name]) for name in PARAMETER_NAMES}


def make_objective(curve, times, datasets):
    jax_times = jnp.asarray(times)

    def objective(raw):
        magnification = curve(jax_times, unpack(raw))
        total = 0.0
        for dataset in datasets:
            total += profile_flux(
                magnification[dataset["slice"]], dataset["flux"], dataset["error"]
            )[0]
        return total

    return objective


def flux_results(curve, times, datasets, parameters):
    magnification = np.asarray(curve(jnp.asarray(times), parameters))
    results = []
    for dataset in datasets:
        chi2, source_flux, blend_flux = profile_flux(
            jnp.asarray(magnification[dataset["slice"]]),
            dataset["flux"],
            dataset["error"],
        )
        results.append((float(chi2), float(source_flux), float(blend_flux)))
    return results


def legacy_best(parameters, results):
    head = [
        parameters["t0"], parameters["tE"], parameters["u0"],
        parameters["q"], parameters["s"], parameters["alpha"], parameters["rho"],
    ]
    head.extend([0.0] * 11)
    tail = []
    for _, source_flux, blend_flux in results:
        tail.extend([source_flux, blend_flux, 0.0, 0.0])
    return np.asarray(head + tail)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--list", required=True, type=Path, dest="list_path")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--time-window", type=float, default=300.0)
    parser.add_argument("--data-center", type=float,
                        help="fixed center used only for selecting data")
    parser.add_argument("--q-init", type=float, default=1.0e-3)
    parser.add_argument("--s-init", type=float, default=1.0)
    parser.add_argument("--alpha-init", type=float, default=0.0)
    parser.add_argument("--rho-init", type=float, default=2.0e-3)
    parser.add_argument("--tol", type=float, default=1.0e-4)
    parser.add_argument("--reltol", type=float, default=1.0e-4)
    parser.add_argument("--max-source-bins", type=int, default=400)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--skip-native-check", action="store_true")
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument(
        "--trust-scale", type=float, default=1.0,
        help="scale local bounds around the warmup geometry",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    defaults = {
        "t0": 0.0, "tE": 1.0, "u0": 0.1, "q": args.q_init,
        "s": args.s_init, "alpha": args.alpha_init, "rho": args.rho_init,
    }
    initial = read_best(args.model, defaults)
    data_center = initial["t0"] if args.data_center is None else args.data_center
    datasets, times = read_data(args.list_path, data_center, args.time_window)
    options = lcbinint.Options(
        jax=True,
        coordinates="vbm_center_of_mass",
        tol=args.tol,
        reltol=args.reltol,
        max_source_bins=args.max_source_bins,
    )
    curve = lcbinint.LightCurve(options=options)
    report = curve.warmup(times, initial)
    if not report.all_calibrated:
        raise RuntimeError("lcbinint JAX warmup did not calibrate every epoch")

    objective = make_objective(curve, times, datasets)
    value_gradient = jax.jit(jax.value_and_grad(objective))
    initial_raw = pack(initial)
    initial_chi2, initial_gradient = value_gradient(jnp.asarray(initial_raw))
    initial_chi2 = float(initial_chi2)
    initial_gradient = np.asarray(initial_gradient, dtype=float)
    if not np.isfinite(initial_chi2) or not np.all(np.isfinite(initial_gradient)):
        raise RuntimeError("Initial JAX chi2 or gradient is non-finite")

    native_delta = np.nan
    if not args.skip_native_check:
        native_curve = lcbinint.LightCurve(
            options=lcbinint.Options(
                coordinates="vbm_center_of_mass", tol=args.tol, reltol=args.reltol,
                max_source_bins=args.max_source_bins,
            )
        )
        jax_magnification = np.asarray(curve(jnp.asarray(times), initial))
        native_magnification = np.asarray(native_curve(times, initial))
        native_delta = float(np.max(np.abs(jax_magnification - native_magnification)))

    if args.evaluate_only:
        fit_raw = initial_raw
        fit_chi2 = initial_chi2
        success = True
        iterations = 0
        message = "evaluate-only"
    else:
        scale = args.trust_scale
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("--trust-scale must be finite and positive")
        # The retained lcbinint warmup plan is local in source position,
        # lens geometry, and source radius. Later stages can re-warm.
        widths = np.asarray(
            [0.02, 0.04, 0.005, 0.04, 0.04, 0.04, np.log(1.25)]
        ) * scale
        lower = initial_raw - widths
        upper = initial_raw + widths
        lower[2] = max(lower[2], -5.0)
        upper[2] = min(upper[2], 5.0)
        lower[3] = max(lower[3], np.log(1.0e-8))
        upper[3] = min(upper[3], np.log(1.0))
        lower[4] = max(lower[4], np.log(0.03))
        upper[4] = min(upper[4], np.log(30.0))
        lower[6] = max(lower[6], np.log(1.0e-6))
        upper[6] = min(upper[6], np.log(1.0))
        bounds = list(zip(lower, upper))
        best_finite = {
            "raw": initial_raw.copy(), "chi2": initial_chi2, "invalid": 0,
        }

        def scipy_function(raw):
            value, gradient = value_gradient(jnp.asarray(raw))
            value = float(value)
            gradient = np.asarray(gradient, dtype=float)
            if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
                best_finite["invalid"] += 1
                return 1.0e100, np.zeros_like(raw)
            if value < best_finite["chi2"]:
                best_finite["raw"] = np.asarray(raw, dtype=float).copy()
                best_finite["chi2"] = value
            return value, gradient

        fit = minimize(
            scipy_function, initial_raw, jac=True, method="L-BFGS-B",
            bounds=bounds, options={"maxiter": args.maxiter},
        )
        fit_raw = np.asarray(fit.x, dtype=float)
        fit_chi2 = float(fit.fun)
        if (not np.isfinite(fit_chi2) or fit_chi2 >= 1.0e99) and np.isfinite(
            best_finite["chi2"]
        ):
            fit_raw = best_finite["raw"]
            fit_chi2 = best_finite["chi2"]
        success = bool(fit.success)
        iterations = int(fit.nit)
        message = f"{fit.message}; invalid_trials={best_finite['invalid']}"
        if not np.isfinite(fit_chi2):
            raise RuntimeError("Binary-lens optimization failed with non-finite chi2")

    fitted = physical(fit_raw)
    results = flux_results(curve, times, datasets, fitted)
    ndata = len(times)
    nparam = 7 + 2 * len(datasets)
    lines = [
        "# generated by jax_binary_static_fit.py",
        "# lens= binary finite_source= uniform engine= autodiff-vbm-JAX",
        "# coordinates= center_of_mass",
        f"# warmup_all_calibrated= {int(report.all_calibrated)}",
        f"# initial_chi2= {initial_chi2:.12e}",
        "# initial_gradient_raw= " + " ".join(f"{value:.12e}" for value in initial_gradient),
        f"# native_jax_max_abs_delta= {native_delta:.12e}",
    ]
    for index, (dataset, result) in enumerate(zip(datasets, results)):
        dataset_times = dataset["time"]
        lines.append(
            f"#Ndata= {len(dataset_times):5d} read from {dataset['path']}, "
            f"JD1= {dataset_times.min():.12f} JD2= {dataset_times.max():.12f}"
        )
        lines.append(
            f"#chi2band {index} {result[0]:.12e} "
            f"{dataset['observatory']} {dataset['band']}"
        )
    lines.append(
        f"#chi2/dof= {fit_chi2:.12e} Ndata= {ndata} nparam= {nparam} "
        f"redchi2= {fit_chi2/(ndata-nparam):.12e}"
    )
    lines.append("#BEST " + " ".join(f"{value:.12e}" for value in legacy_best(fitted, results)))
    lines.append(
        f"#Converged JAX L-BFGS-B success= {int(success)} iterations= {iterations} "
        f"message= {message}"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"chi2={fit_chi2:.12f}")
    print("parameters=" + " ".join(f"{name}={fitted[name]:.12g}" for name in PARAMETER_NAMES))
    print(f"native_jax_max_abs_delta={native_delta:.6e}")
    print(args.output)


if __name__ == "__main__":
    main()
