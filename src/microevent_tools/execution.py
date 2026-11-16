"""Subprocess execution, stage fingerprints and resumable completion records."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from .config import Settings


@dataclass(frozen=True)
class Step:
    name: str
    command: tuple[str, ...]
    outputs: tuple[Path, ...] = ()
    inputs: tuple[Path, ...] = ()
    condition: str = "always"
    accepted_codes: tuple[int, ...] = (0,)


def kernel(settings: Settings, name: str, *args) -> tuple[str, ...]:
    return (settings.python, "-m", f"microevent_tools.kernels.{name}", *map(str, args))


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def environment(settings):
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[key] = str(settings.slots)
    env["PYTHONUNBUFFERED"] = "1"
    # Include this installed/source package for workers, including editable installs.
    additions = [str(Path(__file__).resolve().parents[1])]
    if settings.autodiff_vbm:
        additions.append(str(settings.autodiff_vbm / "src"))
    env["PYTHONPATH"] = os.pathsep.join(additions + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


class Runner:
    def __init__(self, settings, output, base_inputs, *, force=False):
        self.settings = settings
        self.output = output
        self.force = force
        self.base_inputs = tuple(base_inputs)
        self.state_path = output / "pipeline-state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {"steps": {}}
        self.binary_passed = None

    def _write(self):
        self.output.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=2) + "\n")
        temporary.replace(self.state_path)

    def fingerprint(self, step):
        inputs = [self.settings.config_file, *self.base_inputs, *step.inputs]
        hashes = {str(p): file_hash(p) for p in inputs if p.is_file()}
        missing = [str(p) for p in inputs if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"Stage {step.name} inputs missing: {missing}")
        # Source changes invalidate cached work; installed packages ship these .py files.
        code = Path(__file__).parent
        hashes.update({str(p.relative_to(code)): file_hash(p) for p in sorted(code.rglob("*.py"))})
        return hashlib.sha256(json.dumps([step.command, hashes], sort_keys=True).encode()).hexdigest()

    def run(self, step):
        if step.condition == "binary" and self.binary_passed is not True:
            print(f"SKIP {step.name}: binary improvement gate did not pass", flush=True)
            return
        if step.condition == "point_only" and self.binary_passed is not False:
            return
        fingerprint = self.fingerprint(step)
        old = self.state["steps"].get(step.name)
        cached = old and old.get("status") == "complete" and old.get("fingerprint") == fingerprint
        cached = cached and all(p.is_file() and old.get("output_hashes", {}).get(str(p)) == file_hash(p) for p in step.outputs)
        if cached and not self.force:
            print(f"REUSE {step.name}", flush=True)
            if step.name == "binary_gate":
                self.binary_passed = old["returncode"] == 0
            return
        if not self.force and old and old.get("status") == "complete":
            raise RuntimeError(f"Inputs/code/output changed for {step.name}; use a new version or --force")
        if not old and any(p.exists() for p in step.outputs):
            raise FileExistsError(f"Untracked existing outputs for {step.name}; use a new version")
        log_dir = self.output / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        for path in step.outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
        self.state["steps"][step.name] = {"status": "running", "fingerprint": fingerprint}
        self._write()
        command = list(step.command)
        if os.name == "posix":
            command = ["nice", "-n", str(self.settings.nice), *command]
        print(f"RUN {step.name}: {shlex.join(command)}", flush=True)
        start = time.monotonic()
        with (log_dir / f"{step.name}.log").open("a") as log:
            log.write("\n$ " + shlex.join(command) + "\n"); log.flush()
            result = subprocess.run(command, cwd=self.settings.project_root,
                                    env=environment(self.settings), stdout=log, stderr=subprocess.STDOUT)
        valid = result.returncode in step.accepted_codes and all(p.is_file() for p in step.outputs)
        record = self.state["steps"][step.name]
        record.update(status="complete" if valid else "failed", returncode=result.returncode,
                      seconds=time.monotonic()-start,
                      output_hashes={str(p): file_hash(p) for p in step.outputs if p.is_file()})
        self._write()
        if not valid:
            raise RuntimeError(f"Stage {step.name} failed; inspect {log_dir / (step.name + '.log')}")
        if step.name == "binary_gate":
            self.binary_passed = result.returncode == 0
