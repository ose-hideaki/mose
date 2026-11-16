"""Versioned artifact paths, independent of a particular filesystem."""
from dataclasses import dataclass
from pathlib import Path
from .config import Settings


@dataclass(frozen=True)
class EventPaths:
    data_list: Path
    initial: Path
    radec: Path
    output: Path
    model: str

    @classmethod
    def from_settings(cls, config: Settings):
        event = config.project_root / "data" / config.year / config.event
        data = event / config.version
        return cls(data / "list", data / config.model, event / "RADec",
                   config.project_root / "analysis" / config.year / config.event / config.version, config.model)

    def point(self, name: str, post=False) -> Path:
        prefix = "POST_BINARY." if post else ""
        return self.output / "result" / f"{self.model}.{prefix}{name}.VBM"

    @property
    def binary(self):
        return self.output / "result" / "binary" / f"{self.model}.Binary.VBM"

    @property
    def final(self):
        return self.output / "result" / f"{self.model}.Binary.VBM.final"

    @property
    def modes(self):
        return self.output / "grid" / "modes"

    @property
    def summary(self):
        return self.output / "result" / f"{self.model}.MODEL_SUMMARY.tsv"


def suffix(path: Path, value: str) -> Path:
    return Path(str(path) + value)
