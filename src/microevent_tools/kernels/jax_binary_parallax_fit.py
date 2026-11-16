#!/usr/bin/env python3
"""Fit annual parallax independently around one optimized binary-lens mode."""

from __future__ import annotations

# Support both installed modules and the internal subprocess entry points.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "microevent_tools.kernels"


import argparse
import math
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

import os
from . import autodiff_vbm_backend as lcbinint
from .jax_binary_static_fit import flux_results, legacy_best, profile_flux, read_best, read_data

jax.config.update("jax_enable_x64", True)

NAMES = ("t0", "tE", "u0", "q", "s", "alpha", "rho", "pai1", "pai2")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--list", required=True, type=Path, dest="list_path")
    parser.add_argument("--radec", type=Path)
    parser.add_argument(
        "--point-parallax-model", type=Path,
        help=(
            "FSPL-parallax model supplying t0, tE, u0, rho, pai1 and pai2; "
            "binary q, s and alpha still come from --model"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--time-window", type=float, default=300.0)
    parser.add_argument("--data-center", type=float,
                        help="fixed center used only for selecting data")
    parser.add_argument("--max-stages", type=int, default=10)
    parser.add_argument("--delta-chi2", type=float, default=1e-3)
    parser.add_argument("--relative-chi2", type=float, default=1e-7)
    parser.add_argument("--pi-bound", type=float, default=2.0)
    return parser.parse_args()


def read_radec(path: Path) -> tuple[str, str]:
    text = path.read_text()
    match = re.search(r"\bRA=\s*(\S+)\s+Dec=\s*(\S+)", text)
    if match is None:
        raise ValueError(f"Cannot read RA/Dec from {path}")
    return match.group(1), match.group(2)


def static_chi2(path: Path) -> float:
    for line in path.read_text().splitlines():
        if line.startswith("#chi2/dof="):
            return float(line.split()[1])
    raise ValueError(f"No #chi2/dof line in {path}")


def read_reference_time(path: Path, default: float) -> float:
    for line in path.read_text().splitlines():
        if line.startswith("# t_ref="):
            return float(line.split()[2])
    return default


def read_parallax(path: Path) -> tuple[float, float]:
    for line in path.read_text().splitlines():
        if line.startswith("#BEST"):
            values = [float(value) for value in line.split()[1:]]
            if len(values) >= 10:
                return values[8], values[9]
    raise ValueError(f"No pai1/pai2 fields in {path}")


def to_internal(parameters):
    result = dict(parameters)
    result.setdefault("pai1", 0.0)
    result.setdefault("pai2", 0.0)
    return result


def to_output(parameters):
    return dict(parameters)


def pack(parameters):
    return np.asarray([
        parameters["t0"], np.log(parameters["tE"]), parameters["u0"],
        np.log(parameters["q"]), np.log(parameters["s"]), parameters["alpha"],
        np.log(parameters["rho"]), parameters["pai1"], parameters["pai2"],
    ])


def unpack(raw):
    return {
        "t0": raw[0], "tE": jnp.exp(raw[1]), "u0": raw[2],
        "q": jnp.exp(raw[3]), "s": jnp.exp(raw[4]), "alpha": raw[5],
        "rho": jnp.exp(raw[6]), "pai1": raw[7], "pai2": raw[8],
    }


def physical(raw):
    values = unpack(jnp.asarray(raw))
    return {name: float(values[name]) for name in NAMES}


def make_curve(sky, reference_time):
    return lcbinint.LightCurve(
        model=lcbinint.Model(parallax=True, sky=sky, t_ref=reference_time),
        options=lcbinint.Options(
            jax=True, coordinates="vbm_center_of_mass", tol=1e-4,
            reltol=1e-4, max_source_bins=400,
        ),
    )


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


def local_fit(value_gradient, initial, bounds):
    best = {"raw": initial.copy(), "chi2": float("inf"), "invalid": 0}

    def function(raw):
        value, gradient = value_gradient(jnp.asarray(raw))
        value = float(value)
        gradient = np.asarray(gradient, dtype=float)
        if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
            best["invalid"] += 1
            return 1e100, np.zeros_like(raw)
        if value < best["chi2"]:
            best.update(raw=np.asarray(raw).copy(), chi2=value)
        return value, gradient

    fit = minimize(function, initial, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 300})
    if np.isfinite(fit.fun) and fit.fun < best["chi2"]:
        return np.asarray(fit.x), float(fit.fun), best["invalid"]
    return best["raw"], best["chi2"], best["invalid"]


def main() -> None:
    args = parse_args()
    radec = args.radec or args.list_path.parent.parent / "RADec"
    if not radec.is_file():
        raise FileNotFoundError(f"RA/Dec file does not exist: {radec}")
    ra, dec = read_radec(radec)
    sky = lcbinint.obs.SkyCoord(ra, dec)
    defaults = {"t0": 0.0, "tE": 1.0, "u0": 0.1, "q": 1e-3, "s": 1.0, "alpha": 0.0, "rho": 2e-3}
    initial = to_internal(read_best(args.model, defaults))
    reference_time = initial["t0"]
    if args.point_parallax_model is not None:
        if not args.point_parallax_model.is_file():
            raise FileNotFoundError(args.point_parallax_model)
        point = read_best(args.point_parallax_model, defaults)
        point["pai1"], point["pai2"] = read_parallax(args.point_parallax_model)
        for name in ("t0", "tE", "u0", "rho", "pai1", "pai2"):
            initial[name] = point[name]
        reference_time = read_reference_time(args.point_parallax_model, point["t0"])
    data_center = initial["t0"] if args.data_center is None else args.data_center
    datasets, times = read_data(args.list_path, data_center, args.time_window)
    input_static_value = static_chi2(args.model)
    curve = make_curve(sky, reference_time)
    objective = make_objective(curve, times, datasets)
    full_value_gradient = jax.jit(jax.value_and_grad(objective))

    # First locate piE while holding the seven static parameters fixed.
    geometry_raw = pack(initial)
    static_value = float(objective(jnp.asarray(geometry_raw)))
    fixed_geometry = jnp.asarray(geometry_raw[:7])

    def pi_objective(pi_raw):
        return objective(jnp.concatenate((fixed_geometry, pi_raw)))

    pi_value_gradient = jax.jit(jax.value_and_grad(pi_objective))
    pi_candidates = []
    for pi_n, pi_e in ((0.0, 0.0), (0.1, 0.0), (-0.1, 0.0), (0.0, 0.1), (0.0, -0.1)):
        start = geometry_raw.copy()
        start[7:] = (pi_n, pi_e)
        pi_bounds = [
            (max(-args.pi_bound, value - 0.2), min(args.pi_bound, value + 0.2))
            for value in (pi_n, pi_e)
        ]
        pi_fit, value, invalid = local_fit(
            pi_value_gradient, start[7:], pi_bounds
        )
        candidate = start.copy()
        candidate[7:] = pi_fit
        pi_candidates.append((value, candidate, invalid))
    if not pi_candidates:
        raise RuntimeError("No finite parallax seed")
    _, current, invalid_total = min(pi_candidates, key=lambda item: item[0])

    # Then release all nine parameters in repeated local trust stages. Dynamic
    # JAX trajectories route each epoch directly; fixed static warmup plans do
    # not support parallax and therefore must not be used here.
    widths = np.asarray([0.02, 0.04, 0.005, 0.04, 0.04, 0.04, np.log(1.25), 0.10, 0.10])
    previous_chi2 = None
    stages = 0
    for stage in range(1, args.max_stages + 1):
        stages = stage
        lower, upper = current - widths, current + widths
        lower[2], upper[2] = max(lower[2], -5.0), min(upper[2], 5.0)
        lower[7:] = np.maximum(lower[7:], -args.pi_bound)
        upper[7:] = np.minimum(upper[7:], args.pi_bound)
        fitted, value, invalid = local_fit(full_value_gradient, current, list(zip(lower, upper)))
        invalid_total += invalid
        current = fitted
        if previous_chi2 is not None:
            improvement = previous_chi2 - value
            threshold = max(args.delta_chi2, args.relative_chi2 * abs(previous_chi2))
            if improvement >= 0.0 and improvement <= threshold:
                break
        previous_chi2 = value

    fitted_internal = physical(current)
    fitted = to_output(fitted_internal)
    results = flux_results(curve, times, datasets, fitted_internal)
    parallax_value = sum(result[0] for result in results)
    ndata = len(times)
    nparam = 9 + 2 * len(datasets)
    legacy = legacy_best(fitted, results)
    legacy[5] = fitted["alpha"]
    legacy[8] = fitted["pai1"]
    legacy[9] = fitted["pai2"]
    lines = [
        "# generated by jax_binary_parallax_fit.py",
        "# lens= binary finite_source= uniform parallax= annual engine= autodiff-vbm-JAX",
        "# coordinates_internal= vbm_center_of_mass coordinates_output= center_of_mass",
        "# dynamic_trajectory= 1 fixed_warmup= unsupported-not-used",
        f"# t_ref= {reference_time:.12e}",
        f"# sky= RA {ra} Dec {dec}",
        f"# point_parallax_seed= {args.point_parallax_model or 'none'}",
        f"#INPUT_STATIC_CHI2= {input_static_value:.12e}",
        f"#STATIC_CHI2_SAME_WINDOW= {static_value:.12e}",
        f"#PARALLAX_CHI2= {parallax_value:.12e}",
        f"#DELTA_CHI2_PARALLAX_MINUS_STATIC= {parallax_value-static_value:.12e}",
    ]
    for index, (dataset, result) in enumerate(zip(datasets, results)):
        lines.append(f"#chi2band {index} {result[0]:.12e} {dataset['observatory']} {dataset['band']}")
    lines += [
        f"#chi2/dof= {parallax_value:.12e} Ndata= {ndata} nparam= {nparam} redchi2= {parallax_value/(ndata-nparam):.12e}",
        "#BEST " + " ".join(f"{value:.12e}" for value in legacy),
        f"#Converged JAX parallax stages= {stages} invalid_trials= {invalid_total}",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"static_chi2={static_value:.12f}")
    print(f"parallax_chi2={parallax_value:.12f}")
    print(f"delta_chi2={parallax_value-static_value:.12f}")
    print("parameters=" + " ".join(f"{name}={fitted[name]:.12g}" for name in NAMES))
    print(args.output)


if __name__ == "__main__":
    main()
