#!/usr/bin/env python3
"""Decide whether a refined static-binary solution merits further analysis."""

from __future__ import annotations

# Support both installed modules and the internal subprocess entry points.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "microevent_tools.kernels"


import argparse
import math
import sys
from pathlib import Path


SKIP_EXIT_STATUS = 20


def read_fit(path: Path) -> tuple[float, float, int | None, int | None]:
    chi2 = redchi2 = math.nan
    ndata = nparam = None
    for line in path.read_text().splitlines():
        if not line.startswith("#chi2/dof="):
            continue
        fields = line.split()
        chi2 = float(fields[1])
        for index, field in enumerate(fields):
            if field == "Ndata=":
                ndata = int(fields[index + 1])
            elif field == "nparam=":
                nparam = int(fields[index + 1])
            elif field == "redchi2=":
                redchi2 = float(fields[index + 1])
        break
    if not math.isfinite(chi2) or not math.isfinite(redchi2):
        raise ValueError(f"No usable #chi2/dof line in {path}")
    return chi2, redchi2, ndata, nparam


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fspl", required=True, type=Path)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--always-continue", action="store_true",
        help="record the natural decision but return success even when it is skip",
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if not math.isfinite(args.threshold) or args.threshold < 0.0:
        raise ValueError("--threshold must be finite and non-negative")
    fspl_chi2, fspl_redchi2, fspl_ndata, fspl_nparam = read_fit(args.fspl)
    binary_chi2, binary_redchi2, binary_ndata, binary_nparam = read_fit(args.binary)
    if fspl_ndata is not None and binary_ndata is not None and fspl_ndata != binary_ndata:
        raise ValueError(
            f"Ndata mismatch: FSPL has {fspl_ndata}, binary has {binary_ndata}"
        )
    improvement = fspl_redchi2 - binary_redchi2
    natural_decision = "continue" if improvement > args.threshold else "skip"
    decision = (
        "continue_forced"
        if args.always_continue and natural_decision == "skip"
        else natural_decision
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "decision\tthreshold\timprovement\tfspl_chi2\tfspl_redchi2\tfspl_ndata\t"
        "fspl_nparam\tbinary_chi2\tbinary_redchi2\tbinary_ndata\tbinary_nparam\t"
        "fspl_path\tbinary_path\n"
        f"{decision}\t{args.threshold:.12e}\t{improvement:.12e}\t"
        f"{fspl_chi2:.12e}\t{fspl_redchi2:.12e}\t{fspl_ndata}\t{fspl_nparam}\t"
        f"{binary_chi2:.12e}\t{binary_redchi2:.12e}\t{binary_ndata}\t"
        f"{binary_nparam}\t{args.fspl}\t{args.binary}\n"
    )
    print(
        f"binary improvement check: fspl_redchi2={fspl_redchi2:.12g} "
        f"binary_redchi2={binary_redchi2:.12g} improvement={improvement:.12g} "
        f"threshold={args.threshold:.12g} decision={decision}",
        flush=True,
    )
    if natural_decision == "skip" and not args.always_continue:
        sys.exit(SKIP_EXIT_STATUS)


if __name__ == "__main__":
    main()
