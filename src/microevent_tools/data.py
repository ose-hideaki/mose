"""Photometry loading with explicit, unambiguous relative-path resolution."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class Dataset:
    path: Path
    observatory: str
    band: str
    values: np.ndarray


def load_datasets(list_path: Path, center=None, window=None) -> list[Dataset]:
    list_path = list_path.expanduser().resolve()
    datasets = []
    seen = set()
    for number, line in enumerate(list_path.read_text().splitlines(), 1):
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) < 3:
            raise ValueError(f"{list_path}:{number}: expected path observatory band")
        raw = Path(fields[0]).expanduser()
        if raw.is_absolute():
            candidates = [raw.resolve()]
        else:
            candidates = list(dict.fromkeys([(list_path.parent / raw).resolve(), (list_path.parent.parent / raw).resolve()]))
        existing = [p for p in candidates if p.is_file()]
        if len(existing) != 1:
            raise ValueError(f"Missing or ambiguous photometry path on list line {number}: {fields[0]}")
        path = existing[0]
        if path in seen:
            raise ValueError(f"Dataset listed twice: {path}")
        seen.add(path)
        values = np.loadtxt(path, usecols=(0, 1, 2), ndmin=2)
        good = np.isfinite(values).all(axis=1) & (values[:, 2] > 0)
        if center is not None and window is not None:
            good &= np.abs(values[:, 0] - center) <= window
        if good.any():
            datasets.append(Dataset(path, fields[1], fields[2], values[good]))
    if not datasets:
        raise ValueError(f"No usable data in {list_path}")
    return datasets


def read_best(path: Path, defaults: dict | None = None) -> dict:
    names = ("t0", "tE", "u0", "q", "s", "alpha", "rho")
    for line in path.read_text().splitlines():
        if line.startswith("#BEST"):
            values = [float(x) for x in line.split()[1:]]
            if len(values) < 3 or not np.isfinite(values[:7]).all():
                break
            result = dict(defaults or {})
            result.update(zip(names, values[:7]))
            for key in ("tE", "q", "s", "rho"):
                if key in result and result[key] <= 0 and defaults and key in defaults:
                    result[key] = defaults[key]
            return result
    raise ValueError(f"No usable #BEST line: {path}")
