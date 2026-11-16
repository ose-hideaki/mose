#!/usr/bin/env python3
"""Select the minimum-chi2 model across static and parallax binary ranks."""

from __future__ import annotations

# Support both installed modules and the internal subprocess entry points.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "microevent_tools.kernels"


import argparse
import re
import shutil
from pathlib import Path


RANK_PATTERN = re.compile(r"\.rank(\d+)$")


def parse_chi2(path: Path) -> float:
    for line in path.read_text().splitlines():
        if line.startswith("#chi2/dof="):
            return float(line.split()[1])
    raise ValueError(f"No #chi2/dof line in {path}")


def parse_best(path: Path) -> list[float]:
    for line in path.read_text().splitlines():
        if line.startswith("#BEST"):
            return [float(value) for value in line.split()[1:]]
    raise ValueError(f"No #BEST line in {path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--max-rank", type=int,
                        help="Ignore static/parallax models above this rank")
    parser.add_argument("--parallax-suffix", default=".PARALLAX.JAX")
    parser.add_argument("--static-suffix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = []
    for static_path in args.prefix.parent.glob(f"{args.prefix.name}.rank*"):
        match = RANK_PATTERN.search(static_path.name)
        if match is None or not static_path.is_file() or static_path.stat().st_size == 0:
            continue
        rank = int(match.group(1))
        if args.max_rank is not None and rank > args.max_rank:
            continue
        paths = [("static", Path(f"{static_path}{args.static_suffix}"))]
        parallax_path = Path(f"{static_path}{args.parallax_suffix}")
        if parallax_path.is_file() and parallax_path.stat().st_size > 0:
            paths.append(("parallax", parallax_path))
        for model_type, path in paths:
            try:
                values = parse_best(path)
                value_chi2 = parse_chi2(path)
            except (OSError, ValueError):
                continue
            candidates.append(
                {
                    "rank": rank, "type": model_type, "path": path,
                    "chi2": value_chi2, "values": values,
                }
            )
    if not candidates:
        raise RuntimeError(f"No usable ranked model found for prefix {args.prefix}")
    candidates.sort(key=lambda item: (item["chi2"], item["rank"], item["type"]))
    selected = candidates[0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(selected["path"], args.output)

    lines = [
        "# selected_by=min_chi2",
        f"# selected_rank={selected['rank']} selected_type={selected['type']} selected_chi2={selected['chi2']:.12e}",
        f"# selected_source={selected['path']}",
        "rank\ttype\tchi2\tdelta_from_best\tt0\ttE\tu0\tq\ts\talpha\trho\tpai1\tpai2\tpath",
    ]
    for item in candidates:
        values = item["values"]
        pi_en = values[8] if len(values) > 9 else 0.0
        pi_ee = values[9] if len(values) > 9 else 0.0
        core = values[:7]
        lines.append(
            f"{item['rank']}\t{item['type']}\t{item['chi2']:.12e}\t"
            f"{item['chi2']-selected['chi2']:.12e}\t"
            + "\t".join(f"{value:.12e}" for value in core)
            + f"\t{pi_en:.12e}\t{pi_ee:.12e}\t{item['path']}"
        )
    args.summary.write_text("\n".join(lines) + "\n")
    print(
        f"selected rank={selected['rank']} type={selected['type']} "
        f"chi2={selected['chi2']:.12f} source={selected['path']} output={args.output}"
    )


if __name__ == "__main__":
    main()
