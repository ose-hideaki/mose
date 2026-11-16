#!/usr/bin/env python3
"""Plot PSPL, static-binary, and parallax-binary best-fit light curves."""

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import os
import lcbinint
from .jax_binary_parallax_fit import make_curve, read_radec
from .jax_binary_static_fit import profile_flux, read_best, read_data

jax.config.update("jax_enable_x64", True)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", required=True, type=Path, dest="list_path")
    parser.add_argument("--radec", required=True, type=Path)
    parser.add_argument("--pspl", required=True, type=Path)
    parser.add_argument("--fspl", required=True, type=Path)
    parser.add_argument("--pspl-parallax", type=Path)
    parser.add_argument("--fspl-parallax", type=Path)
    parser.add_argument("--static", type=Path)
    parser.add_argument("--parallax", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--window", type=float, default=25.0)
    parser.add_argument("--time-window", type=float, default=300.0)
    parser.add_argument("--data-center", type=float,
                        help="fixed center used only for selecting data")
    parser.add_argument("--only-final", action="store_true",
                        help="Plot only FSPL, static binary and parallax binary")
    parser.add_argument("--point-only", action="store_true",
                        help="Plot only PSPL and FSPL; static/parallax inputs are not required")
    return parser.parse_args()


def pspl_magnification(times, parameters):
    tau = (times - parameters["t0"]) / parameters["tE"]
    u = np.sqrt(parameters["u0"] ** 2 + tau ** 2)
    return (u * u + 2.0) / (u * np.sqrt(u * u + 4.0))


def parallax_components(path):
    from .jax_binary_parallax_fit import read_parallax as strict_read
    text = path.read_text()
    if "# parallax_basis= NE" in text:
        return strict_read(path)
    for line in text.splitlines():
        if line.startswith('#BEST'):
            values = [float(x) for x in line.split()[1:]]
            if len(values) >= 10 and any(values[8:10]):
                raise ValueError('Legacy/unlabelled parallax components: use a fresh NE fit')
    return 0., 0.


def read_reference_time(path, default):
    for line in path.read_text().splitlines():
        if line.startswith("# t_ref="):
            return float(line.split()[2])
    return default


def fspl_magnification(times, parameters):
    from .point_backend import PointLens
    vbm = PointLens()
    vbm.Tol = 1e-4
    vbm.RelTol = 1e-4
    theta = jnp.asarray([np.log(abs(parameters["u0"])), np.log(parameters["tE"]),
                         parameters["t0"], np.log(parameters["rho"])])
    return np.asarray(vbm.ESPLLightCurve(theta, jnp.asarray(times))[0])


def point_parallax_magnification(times, parameters, *, finite_source, ra, dec,
                                 reference_time):
    from .point_backend import PointLens

    vbm = PointLens()
    vbm.Tol = 1e-4
    vbm.RelTol = 1e-4

    vbm.SetObjectCoordinates(f"{ra} {dec}")
    vbm.t0_par_fixed = 1
    vbm.t0_par = reference_time
    raw = [parameters["u0"], np.log(parameters["tE"]), parameters["t0"]]
    if finite_source:
        raw.append(np.log(parameters["rho"]))
    raw.extend([parameters["piEN"], parameters["piEE"]])
    function = (vbm.ESPLLightCurveParallax if finite_source
                else vbm.PSPLLightCurveParallax)
    return np.asarray(function(jnp.asarray(raw), jnp.asarray(times))[0])


def main():
    args = arguments()
    defaults = {"t0": 0.0, "tE": 1.0, "u0": 0.1, "q": 1e-3,
                "s": 1.0, "alpha": 0.0, "rho": 2e-3}
    if args.point_only and args.only_final:
        raise ValueError("--point-only and --only-final cannot be used together")
    if not args.point_only and (args.static is None or args.parallax is None):
        raise ValueError("--static and --parallax are required unless --point-only is used")
    pspl = read_best(args.pspl, defaults)
    fspl = read_best(args.fspl, defaults)
    point_parallax = {}
    for title, path in (("PSPL parallax", args.pspl_parallax),
                        ("FSPL parallax", args.fspl_parallax)):
        if path is not None:
            parameters = read_best(path, defaults)
            parameters["piEN"], parameters["piEE"] = parallax_components(path)
            point_parallax[title] = (parameters, path)
    static = parallax = None
    if not args.point_only:
        static = read_best(args.static, defaults)
        parallax = read_best(args.parallax, defaults)
        parallax["piEN"], parallax["piEE"] = parallax_components(args.parallax)
    default_center = fspl["t0"] if static is None else static["t0"]
    data_center = default_center if args.data_center is None else args.data_center
    datasets, all_times = read_data(args.list_path, data_center, args.time_window)

    data_magnifications = {
        "PSPL": pspl_magnification(all_times, pspl),
        "FSPL": fspl_magnification(all_times, fspl),
    }
    parameters = {"PSPL": pspl, "FSPL": fspl}
    if point_parallax:
        ra, dec = read_radec(args.radec)
        for title, (point_parameters, path) in point_parallax.items():
            data_magnifications[title] = point_parallax_magnification(
                all_times, point_parameters,
                finite_source=title.startswith("FSPL"), ra=ra, dec=dec,
                reference_time=read_reference_time(path, point_parameters["t0"]),
            )
            parameters[title] = point_parameters
    if not args.point_only:
        static_curve = lcbinint.LightCurve(
            options=lcbinint.Options(jax=True, coordinates="vbm_center_of_mass",
                                     tol=1e-4, reltol=1e-4, max_source_bins=400)
        )
        report = static_curve.warmup(all_times, static)
        if not report.all_calibrated:
            raise RuntimeError("Static curve warmup failed")
        ra, dec = read_radec(args.radec)
        sky = lcbinint.obs.SkyCoord(ra, dec)
        reference_time = read_reference_time(args.parallax, parallax["t0"])
        parallax_curve = make_curve(sky, reference_time)
        data_magnifications.update({
            "Static binary": np.asarray(static_curve(jnp.asarray(all_times), static)),
            "Parallax binary": np.asarray(parallax_curve(jnp.asarray(all_times), parallax)),
        })
        parameters.update({"Static binary": static, "Parallax binary": parallax})
    if args.only_final:
        selected = ["FSPL"]
        if "FSPL parallax" in data_magnifications:
            selected.append("FSPL parallax")
        selected.extend(["Static binary", "Parallax binary"])
        data_magnifications = {k: data_magnifications[k]
                               for k in selected}
        parameters = {k: parameters[k]
                      for k in selected}
    colors = ["#3569b0", "#dd6b20", "#2f855a", "#9f4fb0",
              "#c53030", "#805ad5", "#319795", "#b7791f"]
    labels = [f"{d['observatory']} {d['band']}" for d in datasets]

    center = fspl["t0"] if static is None else static["t0"]
    dense = np.linspace(center - args.window, center + args.window, 2400)
    dense_models = {
        "PSPL": pspl_magnification(dense, pspl),
        "FSPL": fspl_magnification(dense, fspl),
    }
    for title, (point_parameters, path) in point_parallax.items():
        dense_models[title] = point_parallax_magnification(
            dense, point_parameters, finite_source=title.startswith("FSPL"),
            ra=ra, dec=dec,
            reference_time=read_reference_time(path, point_parameters["t0"]),
        )
    if not args.point_only:
        static_dense_curve = lcbinint.LightCurve(
            options=lcbinint.Options(jax=True, coordinates="vbm_center_of_mass",
                                     tol=1e-4, reltol=1e-4, max_source_bins=400)
        )
        dense_report = static_dense_curve.warmup(dense, static)
        if not dense_report.all_calibrated:
            raise RuntimeError("Dense static curve warmup failed")
        dense_models.update({
            "Static binary": np.asarray(static_dense_curve(jnp.asarray(dense), static)),
            "Parallax binary": np.asarray(parallax_curve(jnp.asarray(dense), parallax)),
        })

    table_dir = args.output.with_suffix("")
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_specs = []
    for model_index, title in enumerate(parameters):
        magnification = data_magnifications[title]
        total_chi2 = 0.0
        rows = []
        for index, dataset in enumerate(datasets):
            slc = dataset["slice"]
            chi2, fs, fb = profile_flux(
                jnp.asarray(magnification[slc]), dataset["flux"], dataset["error"]
            )
            fs, fb, chi2 = float(fs), float(fb), float(chi2)
            total_chi2 += chi2
            time = dataset["time"] - center
            flux = np.asarray(dataset["flux"])
            error = np.asarray(dataset["error"])
            normalized = (flux - fb) / fs
            normalized_error = error / abs(fs)
            residual = (flux - fs * magnification[slc] - fb) / error
            mask = np.abs(time) <= args.window
            rows.append(np.column_stack((time[mask], normalized[mask],
                                         normalized_error[mask], residual[mask],
                                         np.full(mask.sum(), index))))
        slug = title.lower().replace(" ", "_")
        data_file = table_dir / f"{model_index}_{slug}.data.tsv"
        curve_file = table_dir / f"{model_index}_{slug}.curve.tsv"
        np.savetxt(data_file, np.concatenate(rows), header="dt normalized_flux error residual dataset")
        np.savetxt(curve_file, np.column_stack((dense - center, dense_models[title])),
                   header="dt magnification")
        plot_specs.append((title, total_chi2, data_file, curve_file))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for title, total_chi2, data_file, curve_file in plot_specs:
        fig, axes = plt.subplots(4, 1, figsize=(13, 11), sharex=False,
                                 gridspec_kw={"height_ratios": [3, 1, 3, 1]},
                                 constrained_layout=True)
        wide_ax, wide_residual_ax, zoom_ax, residual_ax = axes
        table = np.loadtxt(data_file)
        curve_table = np.loadtxt(curve_file)
        zoom_window = min(args.window, max(2.0, 0.5 * abs(parameters[title]["tE"])))
        hjd_offset = center
        for index, label in enumerate(labels):
            selected = table[:, 4] == index
            wide_ax.errorbar(
                table[selected, 0] + hjd_offset, table[selected, 1], yerr=table[selected, 2],
                fmt=".", ms=3.5, lw=0.35, alpha=0.55,
                color=colors[index], label=label,
            )
            zoom = selected & (np.abs(table[:, 0]) <= zoom_window)
            zoom_ax.errorbar(
                table[zoom, 0] + hjd_offset, table[zoom, 1], yerr=table[zoom, 2],
                fmt=".", ms=3.5, lw=0.35, alpha=0.55, color=colors[index],
            )
            wide_residual_ax.plot(
                table[selected, 0] + hjd_offset, table[selected, 3], ".", ms=3.0,
                alpha=0.5, color=colors[index],
            )
            residual_ax.plot(
                table[zoom, 0] + hjd_offset, table[zoom, 3], ".", ms=3.0,
                alpha=0.5, color=colors[index],
            )
        wide_ax.plot(curve_table[:, 0] + hjd_offset, curve_table[:, 1], color="black", lw=1.4,
                label="best fit")
        zoom_curve = np.abs(curve_table[:, 0]) <= zoom_window
        zoom_ax.plot(curve_table[zoom_curve, 0] + hjd_offset, curve_table[zoom_curve, 1],
                     color="black", lw=1.4, label="best fit")
        wide_ax.set_title(f"{title}   $\\chi^2={total_chi2:.2f}$")
        wide_ax.set_ylabel("Magnification")
        wide_ax.set_ylim(bottom=min(-0.5, wide_ax.get_ylim()[0]))
        wide_ax.legend(ncol=3, fontsize=8, loc="upper right")
        wide_ax.set_xlim(center - args.window, center + args.window)
        wide_residual_ax.axhline(0.0, color="black", lw=0.8)
        wide_residual_ax.set_ylabel("Residual [$\\sigma$]")
        wide_residual_ax.set_ylim(-8, 8)
        wide_residual_ax.set_xlabel("HJD - 2450000")
        wide_residual_ax.set_xlim(center - args.window, center + args.window)
        zoom_ax.set_ylabel("Magnification")
        zoom_ax.set_xlim(center - zoom_window, center + zoom_window)
        zoom_ax.grid(alpha=0.2)
        p = parameters[title]
        text = "t0={t0:.5f}  tE={tE:.3f}  u0={u0:.5g}".format(**p)
        if "binary" in title.lower():
            text += "  q={q:.4g}  s={s:.4g}  rho={rho:.4g}".format(**p)
        zoom_ax.text(0.02, 0.95, text, transform=zoom_ax.transAxes,
                     va="top", fontsize=8)
        residual_ax.axhline(0.0, color="black", lw=0.8)
        residual_ax.set_ylim(-8, 8)
        residual_ax.set_ylabel("Residual [$\\sigma$]")
        residual_ax.set_xlabel("HJD - 2450000")
        residual_ax.set_xlim(center - zoom_window, center + zoom_window)
        residual_ax.set_ylim(-8, 8)
        slug = title.lower().replace(" ", "_")
        output = args.output.with_name(f"{args.output.stem}.{slug}{args.output.suffix}")
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180)
        plt.close(fig)
        print(output)


if __name__ == "__main__":
    main()
