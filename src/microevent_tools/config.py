"""Validated portable configuration; paths are relative to the TOML file."""
from dataclasses import dataclass
from pathlib import Path
import math
import sys
import tomllib


@dataclass(frozen=True)
class Settings:
    config_file: Path
    project_root: Path
    event: str
    year: str
    version: str
    model: str
    stage: str = "point"
    time_window: float | None = None
    warmup: int = 500
    samples: int = 1000
    chains: int = 2
    seed: int = 0
    max_modes: int = 20
    max_stages: int = 10
    keep: int = 10
    delta_chi2: float = 10.
    binary_threshold: float = 1.
    always_continue: bool = False
    post_binary: bool = True
    python: str = sys.executable
    slots: int = 2
    nice: int = 19
    queue: str = ""
    parallel_environment: str = "smp"
    qsub: str = "qsub"
    grid_results: Path | None = None
    grid_parameters: Path | None = None
    grid_expected: int | None = None
    grid_command: tuple[str, ...] = ()


def load(path: Path) -> Settings:
    path = path.expanduser().resolve()
    with path.open("rb") as stream:
        raw = tomllib.load(stream)
    allowed = {
        "event": {"name", "year", "version", "model", "project_root"},
        "pipeline": {"stage", "time_window", "max_modes", "max_stages", "keep", "delta_chi2", "binary_threshold", "always_continue", "post_binary"},
        "mcmc": {"warmup", "samples", "chains", "seed"},
        "runtime": {"python"},
        "scheduler": {"slots", "nice", "queue", "parallel_environment", "qsub"},
        "grid": {"results", "parameters", "expected", "command"},
    }
    for section, values in raw.items():
        if section not in allowed or not isinstance(values, dict):
            raise ValueError(f"Unknown configuration section: {section}")
        unknown = set(values) - allowed[section]
        if unknown:
            raise ValueError(f"Unknown {section} settings: {sorted(unknown)}")
    def resolve(value):
        result = Path(value).expanduser()
        return (path.parent / result).resolve() if not result.is_absolute() else result.resolve()
    event = raw.get("event", {})
    name = event.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("event.name is required")
    year = str(event.get("year", "2026"))
    version = event.get("version", "v1")
    model = event.get("model", name + "-0")
    for value in (name, year, version, model):
        if not isinstance(value, str) or value in ("", ".", "..") or any(c in value for c in "/\\\n\r\0"):
            raise ValueError("Event/year/version/model must be single path components")
    pipeline = raw.get("pipeline", {})
    runtime = raw.get("runtime", {})
    grid = raw.get("grid", {})
    command = grid.get("command", [])
    if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
        raise ValueError("grid.command must be an argument array, not a shell string")
    result = Settings(config_file=path, project_root=resolve(event.get("project_root", ".")),
        event=name, year=year, version=version, model=model, **pipeline,
        **raw.get("mcmc", {}), **raw.get("scheduler", {}),
        python=str(runtime.get("python", sys.executable)),
        grid_results=resolve(grid["results"]) if grid.get("results") else None,
        grid_parameters=resolve(grid["parameters"]) if grid.get("parameters") else None,
        grid_expected=grid.get("expected"), grid_command=tuple(command))
    if result.stage not in ("point", "binary", "all"):
        raise ValueError("pipeline.stage must be point, binary or all")
    for key in ("warmup", "samples", "chains", "slots", "max_modes", "max_stages", "keep"):
        value = getattr(result, key)
        if type(value) is not int or value < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(result.seed) is not int or result.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(result.nice) is not int or not 0 <= result.nice <= 19:
        raise ValueError("nice must lie in 0..19")
    for key in ("always_continue", "post_binary"):
        if type(getattr(result, key)) is not bool:
            raise ValueError(f"{key} must be true or false")
    for key in ("binary_threshold", "delta_chi2"):
        if not math.isfinite(getattr(result, key)) or getattr(result, key) < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
    if result.time_window is not None and (not math.isfinite(result.time_window) or result.time_window <= 0):
        raise ValueError("time_window must be positive")
    if result.grid_expected is not None and (type(result.grid_expected) is not int or result.grid_expected < 1):
        raise ValueError("grid.expected must be a positive integer")
    if result.stage != "point" and result.grid_results is None:
        raise ValueError("Binary/all pipelines need grid.results (existing or externally generated)")
    return result
