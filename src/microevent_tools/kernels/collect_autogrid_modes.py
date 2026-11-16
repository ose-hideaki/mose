#!/usr/bin/env python3
"""Collect autolens grid_fit SUMMARY tables and emit distinct JAX seed models."""

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


HEADER = re.compile(
    r"^SUMMARY \(LM-grid\)\s+s=(\S+)\s+q=(\S+)\s+rho=(\S+)"
)
ROW = re.compile(
    r"^\s*\d+\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$"
)


def angle_distance(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def same_mode(a: dict, b: dict, args) -> bool:
    return (
        abs(math.log(a["s"] / b["s"])) <= args.mode_log_s
        and abs(math.log(a["q"] / b["q"])) <= args.mode_log_q
        and abs(math.log(a["rho"] / b["rho"])) <= args.mode_log_rho
        and angle_distance(a["alpha"], b["alpha"]) <= args.mode_alpha
        and abs(a["u0"] - b["u0"]) <= args.mode_u0
    )


def parse_result(path: Path, ndata: int) -> list[dict]:
    records = []
    grid = None
    for line in path.read_text(errors="replace").splitlines():
        match = HEADER.match(line)
        if match:
            grid = tuple(float(value) for value in match.groups())
            continue
        match = ROW.match(line)
        if grid is None or not match:
            continue
        redchi2, t0, u0, te, alpha_deg = (float(value) for value in match.groups())
        s, q, rho = grid
        record = {
            "source": path, "redchi2": redchi2, "chi2": redchi2 * (ndata - 4),
            "t0": t0, "u0": u0, "tE": te, "alpha": math.radians(alpha_deg),
            "s": s, "q": q, "rho": rho,
        }
        if all(math.isfinite(value) for key, value in record.items()
               if key not in {"source"}) and min(s, q, rho, te) > 0:
            records.append(record)
    return records


def replace_model(template: Path, output: Path, record: dict, ndata: int) -> None:
    lines = template.read_text().splitlines()
    best = (
        f"#BEST {record['t0']:.12e} {record['tE']:.12e} {record['u0']:.12e} "
        f"{record['q']:.12e} {record['s']:.12e} {record['alpha']:.12e} "
        f"{record['rho']:.12e}"
    )
    chi = (
        f"#chi2/dof= {record['chi2']:.12e} Ndata= {ndata} nparam= 4 "
        f"redchi2= {record['redchi2']:.12e}"
    )
    found_best = found_chi = False
    for index, line in enumerate(lines):
        if line.startswith("#BEST"):
            tail = line.split()[8:]
            lines[index] = best + (" " + " ".join(tail) if tail else "")
            found_best = True
        elif line.startswith("#chi2/dof="):
            lines[index] = chi
            found_chi = True
    if not found_best:
        lines.append(best)
    if not found_chi:
        lines.append(chi)
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--ndata", required=True, type=int)
    parser.add_argument("--max-modes", type=int, default=20)
    parser.add_argument("--q-stratified", action="store_true",
                        help="Reserve one best mode per integer log10(q) bin")
    parser.add_argument("--mode-log-s", type=float, default=0.15)
    parser.add_argument("--mode-log-q", type=float, default=0.7)
    parser.add_argument("--mode-log-rho", type=float, default=1.0)
    parser.add_argument("--mode-alpha", type=float, default=0.5)
    parser.add_argument("--mode-u0", type=float, default=0.08)
    args = parser.parse_args()

    records = []
    result_files = sorted(args.result_dir.glob("result_*.dat"))
    for path in result_files:
        records.extend(parse_result(path, args.ndata))
    if not records:
        raise RuntimeError(f"No usable LM-grid SUMMARY rows in {args.result_dir}")
    records.sort(key=lambda item: item["chi2"])
    modes = []
    if args.q_stratified:
        best_by_q_bin = {}
        for record in records:
            q_bin = math.floor(math.log10(record["q"]))
            best_by_q_bin.setdefault(q_bin, record)
        for q_bin in sorted(best_by_q_bin):
            record = best_by_q_bin[q_bin]
            if not any(same_mode(record, mode, args) for mode in modes):
                modes.append(record)
    for record in records:
        if len(modes) == args.max_modes:
            break
        if any(same_mode(record, mode, args) for mode in modes):
            continue
        modes.append(record)

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    aggregate = Path(f"{args.output_prefix}.all.tsv")
    header = "chi2\tredchi2\tt0\tu0\ttE\ts\tq\trho\talpha_rad\tsource"
    aggregate.write_text(header + "\n" + "\n".join(
        f"{r['chi2']:.12e}\t{r['redchi2']:.12e}\t{r['t0']:.12e}\t{r['u0']:.12e}\t"
        f"{r['tE']:.12e}\t{r['s']:.12e}\t{r['q']:.12e}\t{r['rho']:.12e}\t"
        f"{r['alpha']:.12e}\t{r['source']}" for r in records
    ) + "\n")
    candidate_list = Path(f"{args.output_prefix}.candidates")
    paths = []
    for rank, mode in enumerate(modes, 1):
        path = Path(f"{args.output_prefix}.mode{rank}")
        replace_model(args.template, path, mode, args.ndata)
        paths.append(path)
    candidate_list.write_text("\n".join(str(path) for path in paths) + "\n")
    print(f"{len(records)} SUMMARY solutions -> {len(modes)} modes")
    print(f"Candidates: {candidate_list}")
    print(f"Aggregate:  {aggregate}")


if __name__ == "__main__":
    main()
