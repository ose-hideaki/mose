#!/usr/bin/env python3
"""Find distinct close/wide static-binary modes with repeated local JAX fits."""

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
import shutil
import subprocess
import sys
from pathlib import Path


def best_values(path: Path) -> list[float]:
    for line in path.read_text().splitlines():
        if line.startswith("#BEST"):
            values = [float(value) for value in line.split()[1:]]
            if len(values) >= 7:
                return values
    raise ValueError(f"No binary-lens #BEST line in {path}")


def chi2(path: Path) -> float:
    for line in path.read_text().splitlines():
        if line.startswith("#chi2/dof="):
            return float(line.split()[1])
    raise ValueError(f"No #chi2/dof line in {path}")


def angle_distance(first: float, second: float) -> float:
    return abs((first - second + math.pi) % (2.0 * math.pi) - math.pi)


def same_mode(first: list[float], second: list[float], log_s_threshold: float) -> bool:
    return (
        abs(math.log(first[3]) - math.log(second[3])) <= 0.7
        and abs(math.log(first[4]) - math.log(second[4])) <= log_s_threshold
        and angle_distance(first[5], second[5]) <= 0.5
        and abs(math.log(first[6]) - math.log(second[6])) <= 1.0
        and abs(first[2] - second[2]) <= 0.08
    )


def candidate_paths(model: Path, candidates_file: Path | None) -> list[Path]:
    paths = [model]
    if candidates_file is not None:
        base = candidates_file.parent
        for line in candidates_file.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            path = Path(line.split()[0])
            if not path.is_absolute() and not path.exists():
                path = base / path
            if path.is_file():
                paths.append(path)
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def make_seed(source: Path, destination: Path, separation: float) -> None:
    lines = source.read_text().splitlines()
    replaced = False
    for index, line in enumerate(lines):
        if not line.startswith("#BEST"):
            continue
        fields = line.split()
        if len(fields) < 8:
            raise ValueError(f"No binary-lens #BEST line in {source}")
        fields[5] = f"{separation:.12e}"
        lines[index] = " ".join(fields)
        replaced = True
        break
    if not replaced:
        raise ValueError(f"No #BEST line in {source}")
    destination.write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--list", required=True, type=Path, dest="list_path")
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--candidates-file", type=Path)
    parser.add_argument("--max-seed-modes", type=int, default=10)
    parser.add_argument("--fitter", type=Path, default=Path(__file__).with_name("jax_binary_static_fit.py"))
    parser.add_argument("--time-window", type=float, default=300.0)
    parser.add_argument("--data-center", type=float,
                        help="fixed center used only for selecting data")
    parser.add_argument("--max-stages", type=int, default=10)
    parser.add_argument("--min-stages", type=int, default=2)
    parser.add_argument("--delta-chi2", type=float, default=1.0e-3)
    parser.add_argument("--relative-chi2", type=float, default=1.0e-7)
    parser.add_argument("--mode-log-s", type=float, default=0.15)
    parser.add_argument("--keep", type=int, default=10)
    parser.add_argument("--trust-scale", type=float, default=1.0)
    parser.add_argument("--no-reciprocal-seeds", action="store_true",
                        help="Refine only supplied modes; do not add reciprocal-s seeds")
    parser.add_argument("--mcmc", action="store_true")
    parser.add_argument("--mcmc-script", type=Path, default=Path(__file__).with_name("jax_binary_mcmc.py"))
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resume-refinements", action="store_true",
        help="Reuse the highest existing candidateN.stageM file for each candidate",
    )
    return parser.parse_args()


def optimize_seed(args, seed: Path, candidate: int) -> Path:
    previous = seed
    previous_chi2 = None
    for stage in range(1, args.max_stages + 1):
        output = Path(f"{args.output_prefix}.candidate{candidate}.stage{stage}")
        command = [
            sys.executable, str(args.fitter), "--model", str(previous),
            "--list", str(args.list_path), "--output", str(output),
            "--time-window", str(args.time_window), "--trust-scale", str(args.trust_scale),
        ]
        if args.data_center is not None:
            command.extend(["--data-center", str(args.data_center)])
        subprocess.run(command, check=True)
        current_chi2 = chi2(output)
        if previous_chi2 is not None and stage >= args.min_stages:
            improvement = previous_chi2 - current_chi2
            threshold = max(args.delta_chi2, args.relative_chi2 * abs(previous_chi2))
            if improvement >= 0.0 and improvement <= threshold:
                return output
        previous = output
        previous_chi2 = current_chi2
    return previous


def main() -> None:
    args = parse_args()
    if args.keep < 1 or args.max_stages < 1:
        raise ValueError("--keep and --max-stages must be positive")
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    if args.resume_refinements:
        pattern = re.compile(re.escape(str(args.output_prefix)) + r"\.candidate(\d+)\.stage(\d+)$")
        latest: dict[int, tuple[int, Path]] = {}
        for path in args.output_prefix.parent.glob(f"{args.output_prefix.name}.candidate*.stage*"):
            match = pattern.fullmatch(str(path))
            if match is None:
                continue
            candidate, stage = (int(value) for value in match.groups())
            if candidate not in latest or stage > latest[candidate][0]:
                latest[candidate] = (stage, path)
        finals = []
        for candidate in sorted(latest):
            path = latest[candidate][1]
            try:
                values = best_values(path)
                value_chi2 = chi2(path)
            except (OSError, ValueError):
                continue
            if all(math.isfinite(value) for value in values[:7]):
                finals.append({"path": path, "chi2": value_chi2, "values": values})
        if not finals:
            raise RuntimeError(f"No completed refinements found for {args.output_prefix}")
        print(f"reusing {len(finals)} completed candidate refinements")
    else:
        finals = optimize_candidates(args)

    distinct = []
    for result in sorted(finals, key=lambda item: item["chi2"]):
        if any(same_mode(result["values"], item["values"], args.mode_log_s) for item in distinct):
            continue
        distinct.append(result)
        if len(distinct) == args.keep:
            break

    summary = ["# rank chi2 t0 tE u0 q s alpha rho source"]
    for rank, result in enumerate(distinct, start=1):
        ranked = Path(f"{args.output_prefix}.rank{rank}")
        shutil.copyfile(result["path"], ranked)
        values = result["values"][:7]
        summary.append(
            f"{rank} {result['chi2']:.12e} "
            + " ".join(f"{value:.12e}" for value in values)
            + f" {result['path']}"
        )
        print(f"rank{rank}: chi2={result['chi2']:.8f} s={values[4]:.8g} -> {ranked}")
        if args.mcmc:
            command = [
                    sys.executable, str(args.mcmc_script), "--model", str(ranked),
                    "--list", str(args.list_path), "--output", f"{ranked}.mcmc.npz",
                    "--time-window", str(args.time_window), "--warmup", str(args.warmup),
                    "--samples", str(args.samples), "--chains", str(args.chains),
                    "--seed", str(args.seed + rank - 1),
                ]
            if args.data_center is not None:
                command.extend(["--data-center", str(args.data_center)])
            subprocess.run(command, check=True)
    Path(f"{args.output_prefix}.summary").write_text("\n".join(summary) + "\n")
    if not distinct:
        raise RuntimeError("No finite distinct binary-lens solution was found")


def optimize_candidates(args) -> list[dict]:
    raw_candidates = []
    for path in candidate_paths(args.model, args.candidates_file):
        try:
            values = best_values(path)
            value_chi2 = chi2(path)
        except (OSError, ValueError):
            continue
        if (all(math.isfinite(value) for value in values[:7])
                and all(values[index] > 0.0 for index in (1, 3, 4, 6))
                and -5.0 < values[2] < 5.0):
            raw_candidates.append({"path": path, "values": values, "chi2": value_chi2})
    clustered = []
    for candidate in sorted(raw_candidates, key=lambda item: item["chi2"]):
        if any(same_mode(candidate["values"], item["values"], args.mode_log_s) for item in clustered):
            continue
        clustered.append(candidate)
        if len(clustered) == args.max_seed_modes:
            break
    if not clustered:
        raise RuntimeError("No usable grid/refine candidates were found")

    seeds = []
    for candidate in clustered:
        separations = (candidate["values"][4],) if args.no_reciprocal_seeds else (
            candidate["values"][4], 1.0 / candidate["values"][4])
        for separation in separations:
            values = candidate["values"].copy()
            values[4] = separation
            if any(same_mode(values, item["values"], args.mode_log_s) for item in seeds):
                continue
            seeds.append({"source": candidate["path"], "values": values})
    finals = []
    failures = []
    failures_path = Path(f"{args.output_prefix}.failures.tsv")
    for candidate, seed_info in enumerate(seeds, start=1):
        seed = Path(f"{args.output_prefix}.candidate{candidate}.seed")
        try:
            make_seed(seed_info["source"], seed, seed_info["values"][4])
            result = optimize_seed(args, seed, candidate)
            values = best_values(result)
            finals.append({"path": result, "chi2": chi2(result), "values": values})
        except (subprocess.CalledProcessError, OSError, ValueError, RuntimeError) as error:
            message = f"candidate{candidate}\t{seed_info['source']}\t{type(error).__name__}: {error}"
            failures.append(message)
            print(f"WARNING: skipping failed {message}", file=sys.stderr, flush=True)

    if failures:
        failures_path.write_text(
            "candidate\tsource\terror\n" + "\n".join(failures) + "\n"
        )
    elif failures_path.exists():
        failures_path.unlink()

    return finals


if __name__ == "__main__":
    main()
