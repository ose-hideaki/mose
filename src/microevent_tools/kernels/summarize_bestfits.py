#!/usr/bin/env python3
"""Write one comparison table for point- and binary-lens fits."""

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


RANK_PATTERN = re.compile(r"\.rank(\d+)$")
PARAMETERS = ("t0", "tE", "u0", "q", "s", "alpha", "rho", "pai1", "pai2")


def parse_model(path: Path, model: str, rank: int | None = None):
    values = None
    chi2 = redchi2 = math.nan
    ndata = nparam = None
    for line in path.read_text().splitlines():
        if line.startswith("#BEST"):
            values = [float(value) for value in line.split()[1:]]
        elif line.startswith("#chi2/dof="):
            fields = line.split()
            chi2 = float(fields[1])
            for index, field in enumerate(fields):
                if field == "Ndata=":
                    ndata = int(fields[index + 1])
                elif field == "nparam=":
                    nparam = int(fields[index + 1])
                elif field == "redchi2=":
                    redchi2 = float(fields[index + 1])
    if values is None or not math.isfinite(chi2):
        raise ValueError(f"No usable fit result in {path}")
    padded = values + [0.0] * max(0, 10 - len(values))
    parameters = {
        "t0": padded[0], "tE": padded[1], "u0": padded[2],
        "q": padded[3], "s": padded[4], "alpha": padded[5],
        "rho": padded[6], "pai1": padded[8], "pai2": padded[9],
    }
    return {
        "model": model, "rank": rank, "chi2": chi2, "redchi2": redchi2,
        "ndata": ndata, "nparam": nparam, "parameters": parameters, "path": path,
    }


def best_ranked(prefix: Path, parallax: bool, parallax_suffix: str, static_suffix: str = ""):
    candidates = []
    for static_path in prefix.parent.glob(f"{prefix.name}.rank*"):
        match = RANK_PATTERN.search(static_path.name)
        if match is None:
            continue
        rank = int(match.group(1))
        path = Path(f"{static_path}{parallax_suffix}") if parallax else Path(f"{static_path}{static_suffix}")
        if not path.is_file() or path.stat().st_size == 0:
            continue
        try:
            candidates.append(parse_model(path, "binary_parallax" if parallax else "binary_static", rank))
        except (OSError, ValueError):
            continue
    return min(candidates, key=lambda item: item["chi2"]) if candidates else None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pspl", required=True, type=Path)
    parser.add_argument("--fspl", required=True, type=Path)
    parser.add_argument("--pspl-parallax", type=Path)
    parser.add_argument("--fspl-parallax", type=Path)
    parser.add_argument("--binary-prefix", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--parallax-suffix", default=".PARALLAX.JAX")
    parser.add_argument("--static-suffix", default="")
    parser.add_argument("--point-only", action="store_true",
                        help="Write point-lens rows and mark both binary rows unavailable")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = [("PSPL", args.pspl), ("FSPL", args.fspl)]
    if args.pspl_parallax is not None:
        requested.append(("PSPL_PARALLAX", args.pspl_parallax))
    if args.fspl_parallax is not None:
        requested.append(("FSPL_PARALLAX", args.fspl_parallax))
    results = []
    for model, path in requested:
        if path.is_file() and path.stat().st_size > 0:
            try:
                results.append(parse_model(path, model))
            except (OSError, ValueError):
                pass
    if not args.point_only:
        results.extend(
            result for result in (
                best_ranked(args.binary_prefix, False, args.parallax_suffix, args.static_suffix),
                best_ranked(args.binary_prefix, True, args.parallax_suffix),
            ) if result is not None
        )
    by_model = {result["model"].lower(): result for result in results}
    pspl_result = by_model.get("pspl")
    pspl_chi2 = pspl_result["chi2"] if pspl_result is not None else math.nan
    order = (
        "pspl", "pspl_parallax", "fspl", "fspl_parallax",
        "binary_static", "binary_parallax",
    )
    header = [
        "model", "available", "rank", "chi2", "delta_chi2_vs_pspl",
        "redchi2", "ndata", "nparam", *PARAMETERS, "path",
    ]
    lines = ["\t".join(header)]
    for name in order:
        result = by_model.get(name)
        if result is None:
            lines.append("\t".join([name, "0"] + ["NA"] * (len(header) - 2)))
            continue
        same_data = (
            pspl_result is not None and result["ndata"] is not None
            and result["ndata"] == pspl_result["ndata"]
        )
        delta = result["chi2"] - pspl_chi2 if math.isfinite(pspl_chi2) and same_data else math.nan
        fields = [
            name, "1", str(result["rank"] if result["rank"] is not None else "NA"),
            f"{result['chi2']:.12e}", f"{delta:.12e}", f"{result['redchi2']:.12e}",
            str(result["ndata"] if result["ndata"] is not None else "NA"),
            str(result["nparam"] if result["nparam"] is not None else "NA"),
        ]
        fields.extend(f"{result['parameters'][parameter]:.12e}" for parameter in PARAMETERS)
        fields.append(str(result["path"]))
        lines.append("\t".join(fields))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
