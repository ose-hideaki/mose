#!/usr/bin/env python3
"""Fit one microlensing model with native VBMicrolensing and emcee only."""

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

import emcee
import numpy as np
from scipy.optimize import minimize
import VBMicrolensing


PARAMETERS = {
    "pspl": ("t0", "tE", "u0"),
    "fspl": ("t0", "tE", "u0", "rho"),
    "pspl_parallax": ("t0", "tE", "u0", "pai1", "pai2"),
    "fspl_parallax": ("t0", "tE", "u0", "rho", "pai1", "pai2"),
    "binary_static": ("t0", "tE", "u0", "rho", "q", "s", "alpha"),
    "binary_parallax": (
        "t0", "tE", "u0", "rho", "q", "s", "alpha", "pai1", "pai2",
    ),
}


def output_path(prefix: Path, suffix: str) -> Path:
    """Append an output suffix without replacing dots in the prefix name."""
    return prefix.parent / f"{prefix.name}{suffix}"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=PARAMETERS)
    parser.add_argument("--datalist", required=True, type=Path)
    parser.add_argument("--init", required=True, nargs="+", type=float)
    parser.add_argument("--lower", required=True, nargs="+", type=float)
    parser.add_argument("--upper", required=True, nargs="+", type=float)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--coords", help='RA Dec, e.g. "17:55:38 -29:13:22"')
    parser.add_argument("--radec-file", type=Path)
    parser.add_argument("--t0-par", type=float)
    parser.add_argument("--data-center", type=float)
    parser.add_argument("--time-window", type=float, default=400.0)
    parser.add_argument("--vbm-tol", type=float, default=1e-3)
    parser.add_argument(
        "--espl-table", type=Path,
        default=Path(VBMicrolensing.__file__).resolve().parent / "data" / "ESPL.tbl",
    )
    parser.add_argument("--nwalkers", type=int)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--discard", type=int, default=1000)
    parser.add_argument("--thin", type=int, default=10)
    parser.add_argument("--check-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-minimize", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def coordinates(args: argparse.Namespace) -> str | None:
    if args.coords:
        return args.coords
    if args.radec_file:
        import re
        text = args.radec_file.read_text()
        match = re.search(r"\bRA=\s*(\S+)\s+Dec=\s*(\S+)", text)
        if match:
            return f"{match[1]} {match[2]}"
        values = text.split()
        if len(values) < 2:
            raise ValueError(f"invalid RA/Dec file: {args.radec_file}")
        return f"{values[0]} {values[1]}"
    return None


def load_data(path: Path, center: float | None, window: float) -> list[dict]:
    from ..data import load_datasets
    return [{"path": str(d.path), "label": f"{d.observatory} {d.band}",
             "values": d.values} for d in load_datasets(path, center, window)]


def linear_flux(magnification: np.ndarray, flux: np.ndarray, error: np.ndarray):
    weight = 1.0 / np.square(error)
    saa = np.sum(weight * magnification * magnification)
    sa = np.sum(weight * magnification)
    s1 = np.sum(weight)
    say = np.sum(weight * magnification * flux)
    sy = np.sum(weight * flux)
    determinant = saa * s1 - sa * sa
    if not np.isfinite(determinant) or determinant <= 0:
        return np.nan, np.nan, np.inf
    source = (say * s1 - sy * sa) / determinant
    blend = (sy * saa - say * sa) / determinant
    residual = (flux - source * magnification - blend) / error
    return float(source), float(blend), float(np.sum(residual * residual))


class NativeModel:
    def __init__(self, name: str, datasets: list[dict], coords: str | None,
                 t0_par: float | None, tolerance: float, espl_table: Path):
        self.name = name
        self.datasets = datasets
        self.vbm = VBMicrolensing.VBMicrolensing()
        self.vbm.Tol = tolerance
        self.vbm.LoadESPLTable(str(espl_table))
        if "parallax" in name:
            if coords is None:
                raise ValueError("a parallax model requires --coords or --radec-file")
            self.vbm.SetObjectCoordinates(coords)
            if t0_par is not None:
                self.vbm.t0_par_fixed = 1
                self.vbm.t0_par = t0_par

    def magnification(self, theta: np.ndarray, times: np.ndarray) -> np.ndarray:
        p = dict(zip(PARAMETERS[self.name], theta))
        if p["tE"] <= 0 or p["u0"] == 0:
            raise ValueError("invalid point-lens geometry")
        if self.name == "pspl":
            result = self.vbm.PSPLLightCurve(
                [np.log(abs(p["u0"])), np.log(p["tE"]), p["t0"]], times)
        elif self.name == "fspl":
            if p["rho"] <= 0:
                raise ValueError("rho must be positive")
            result = self.vbm.ESPLLightCurve(
                [np.log(abs(p["u0"])), np.log(p["tE"]), p["t0"], np.log(p["rho"])], times)
        elif self.name == "pspl_parallax":
            result = self.vbm.PSPLLightCurveParallax(
                [p["u0"], np.log(p["tE"]), p["t0"], p["pai1"], p["pai2"]], times)
        elif self.name == "fspl_parallax":
            if p["rho"] <= 0:
                raise ValueError("rho must be positive")
            result = self.vbm.ESPLLightCurveParallax([
                p["u0"], np.log(p["tE"]), p["t0"], np.log(p["rho"]),
                p["pai1"], p["pai2"],
            ], times)
        else:
            if p["rho"] <= 0 or p["q"] <= 0 or p["s"] <= 0:
                raise ValueError("rho, q and s must be positive")
            values = [
                np.log(p["s"]), np.log(p["q"]), p["u0"], p["alpha"],
                np.log(p["rho"]), np.log(p["tE"]), p["t0"],
            ]
            if self.name == "binary_parallax":
                values += [p["pai1"], p["pai2"]]
                result = self.vbm.BinaryLightCurveParallax(values, times)
            else:
                result = self.vbm.BinaryLightCurve(values, times)
        return np.asarray(result[0], dtype=float)

    def chi2(self, theta: np.ndarray, *, return_bands: bool = False):
        total = 0.0
        bands = []
        try:
            for dataset in self.datasets:
                data = dataset["values"]
                mag = self.magnification(theta, data[:, 0])
                source, blend, value = linear_flux(mag, data[:, 1], data[:, 2])
                if not np.isfinite(value):
                    raise ValueError("non-finite chi2")
                total += value
                bands.append({
                    "label": dataset["label"], "ndata": len(data),
                    "source_flux": source, "blend_flux": blend, "chi2": value,
                })
        except Exception:
            return (np.inf, []) if return_bands else np.inf
        return (float(total), bands) if return_bands else float(total)


def main() -> None:
    args = arguments()
    names = PARAMETERS[args.model]
    ndim = len(names)
    if not (len(args.init) == len(args.lower) == len(args.upper) == ndim):
        raise ValueError(f"{args.model} requires {ndim} values: {', '.join(names)}")
    initial = np.asarray(args.init, dtype=float)
    lower = np.asarray(args.lower, dtype=float)
    upper = np.asarray(args.upper, dtype=float)
    if np.any(lower >= upper) or np.any(initial <= lower) or np.any(initial >= upper):
        raise ValueError("require lower < init < upper for every parameter")
    datasets = load_data(args.datalist, args.data_center, args.time_window)
    model = NativeModel(
        args.model, datasets, coordinates(args), args.t0_par,
        args.vbm_tol, args.espl_table,
    )

    bounds = list(zip(lower, upper))
    optimum = initial
    if not args.skip_minimize:
        fit = minimize(model.chi2, initial, method="Powell", bounds=bounds)
        if fit.success and np.isfinite(fit.fun):
            optimum = fit.x

    def log_probability(theta):
        if np.any(theta <= lower) or np.any(theta >= upper):
            return -np.inf
        value = model.chi2(theta)
        return -0.5 * value if np.isfinite(value) else -np.inf

    nwalkers = args.nwalkers or max(32, 4 * ndim)
    if nwalkers < 2 * ndim:
        raise ValueError("nwalkers must be at least twice the parameter count")
    rng = np.random.default_rng(args.seed)
    scale = np.maximum((upper - lower) * 1e-3, np.abs(optimum) * 1e-5)
    walkers = []
    while len(walkers) < nwalkers:
        proposal = optimum + rng.normal(size=ndim) * scale
        if np.all(proposal > lower) and np.all(proposal < upper):
            walkers.append(proposal)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
    sampler.random_state = np.random.RandomState(args.seed).get_state()
    previous_tau = None
    stable = 0
    converged = False
    for _ in sampler.sample(
        np.asarray(walkers), iterations=args.steps, progress=not args.no_progress,
    ):
        if sampler.iteration % args.check_every:
            continue
        try:
            tau = sampler.get_autocorr_time(tol=0)
        except Exception:
            continue
        if previous_tau is not None and np.all(np.abs(tau - previous_tau) / tau < 0.01):
            stable += 1
        else:
            stable = 0
        previous_tau = tau
        if stable >= 3 and sampler.iteration > 50 * np.max(tau):
            converged = True
            break

    discard = min(args.discard, max(0, sampler.iteration // 2))
    thin = max(1, args.thin)
    flat = sampler.get_chain(discard=discard, thin=thin, flat=True)
    flat_logp = sampler.get_log_prob(discard=discard, thin=thin, flat=True)
    finite = np.isfinite(flat_logp)
    if not np.any(finite):
        raise RuntimeError("emcee produced no finite samples")
    index = np.flatnonzero(finite)[np.argmax(flat_logp[finite])]
    best = flat[index]
    best_chi2, bands = model.chi2(best, return_bands=True)

    prefix = args.output_prefix.expanduser().resolve()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    chain_path = output_path(prefix, ".chain.npz")
    np.savez_compressed(
        chain_path, chain=sampler.get_chain(),
        log_prob=sampler.get_log_prob(), parameters=np.asarray(names),
    )
    bestfit = output_path(prefix, ".bestfit.tsv")
    bestfit.write_text(
        "parameter\tvalue\n" + "".join(
            f"{name}\t{value:.16g}\n" for name, value in zip(names, best)
        ) + f"chi2\t{best_chi2:.16g}\n"
    )
    summary = {
        "engine": "VBMicrolensing-native", "sampler": "emcee",
        "model": args.model, "parameters": dict(zip(names, map(float, best))),
        "chi2": best_chi2, "ndata": sum(item["ndata"] for item in bands),
        "bands": bands, "steps": sampler.iteration, "nwalkers": nwalkers,
        "discard": discard, "thin": thin, "converged": converged,
        "datalist": str(args.datalist.resolve()),
        "time_window": args.time_window, "data_center": args.data_center,
        "chain": str(chain_path),
        "bestfit": str(bestfit),
    }
    output_path(prefix, ".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
