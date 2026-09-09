"""Atomic rental records and process locks for Vast recovery."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

from ..protocol.models import VastDeploymentRecord
from .config import SETTINGS_DIR

VAST_STATE_DIR = SETTINGS_DIR / "vast"


class VastState:
    """Keep credential-bearing records private and never ignore corrupt state."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or VAST_STATE_DIR

    def directory(self, name: str) -> Path:
        return self.root / hashlib.sha256(name.encode()).hexdigest()[:24]

    @contextmanager
    def locked(self, name: str) -> Iterator[None]:
        import fcntl

        directory = self.directory(name)
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
        with (directory / "lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def load(self, name: str) -> VastDeploymentRecord | None:
        return self._read(self.directory(name) / "record.json")

    def records(self) -> list[VastDeploymentRecord]:
        return [row for path in self.root.glob("*/record.json") if (row := self._read(path)) is not None]

    @staticmethod
    def _read(path: Path) -> VastDeploymentRecord | None:
        try:
            raw = json.loads(path.read_text())
            if not isinstance(raw, dict):
                raise ValueError("Invalid record")
            return VastDeploymentRecord(**raw)
        except FileNotFoundError:
            return None
        except (OSError, TypeError, ValueError):
            raise ValueError(f"Cannot read Vast deployment record {path}; repair it before renting again.") from None

    def save(self, record: VastDeploymentRecord) -> None:
        directory = self.directory(record.name)
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix="record-", dir=directory)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(asdict(record), stream)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(directory / "record.json")
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)

    def remove(self, name: str) -> None:
        (self.directory(name) / "record.json").unlink(missing_ok=True)
