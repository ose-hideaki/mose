"""Adapter for an external autogrid installation; no private map assets bundled."""
import json
from pathlib import Path
import subprocess
import numpy as np
from .data import load_datasets, read_best
from .execution import environment
from .paths import EventPaths


def expected(config):
    if config.grid_expected:
        return config.grid_expected
    if config.grid_parameters is None:
        raise ValueError("Specify grid.expected or grid.parameters to verify completeness")
    values = np.load(config.grid_parameters)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("grid.parameters must contain (logs, logq, logrho) rows")
    return len(np.unique(values[values[:, 1] <= 0], axis=0))


def verify_results(config):
    files = sorted(config.grid_results.glob("result_*.dat"))
    complete = [p for p in files if "SUMMARY (LM-grid)" in p.read_text(errors="replace")]
    count = expected(config)
    if len(complete) != count:
        raise RuntimeError(f"External grid incomplete: {len(complete)}/{count}; rerun after grid completes")
    return complete


def prepare_and_execute(config):
    paths = EventPaths.from_settings(config)
    initial = read_best(paths.initial)
    datasets = load_datasets(paths.data_list, initial["t0"], config.time_window)
    target = paths.output / "grid" / "photometry"
    target.mkdir(parents=True, exist_ok=True)
    entries = []; lines = []
    for i, dataset in enumerate(datasets):
        path = target / f"dataset{i:03d}.flux"
        np.savetxt(path, dataset.values, fmt="%.14e")
        entries.append({"name": f"{dataset.observatory}_{dataset.band}_{i}", "path": str(path)})
        lines.append(f"{path} {dataset.observatory} {dataset.band}")
    grid_json = target.parent / "datasets.json"
    grid_list = target.parent / "list"
    grid_json.write_text(json.dumps({"datasets": entries}, indent=2))
    grid_list.write_text("\n".join(lines) + "\n")
    if config.grid_command:
        pspl = read_best(paths.point("PSPL"))
        values = dict(python=config.python, project_root=str(config.project_root),
            config=str(config.config_file), grid_json=str(grid_json), grid_list=str(grid_list),
            results=str(config.grid_results), qsub_dir=str(target.parent / "qsub"),
            parameters=str(config.grid_parameters or ""), t0=pspl['t0'], tE=pspl['tE'], u0=pspl['u0'])
        command = [argument.format_map(values) for argument in config.grid_command]
        subprocess.run(command, cwd=config.project_root, env=environment(config), check=True)
    # An asynchronous submission may still be running; do not call the pipeline complete.
    verify_results(config)
