"""Best-effort download accounting for Vast's Hugging Face caches."""

from dataclasses import dataclass
from pathlib import PurePosixPath
import re

from huggingface_hub import HfApi

from ..protocol.enums import BackendType
from ..protocol.models import DeploymentConfig

_SPLIT_GGUF = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
_BLOB = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$", re.IGNORECASE)
_PARTIAL_SUFFIXES = (".downloadInProgress", ".incomplete")


@dataclass(frozen=True)
class DownloadFile:
    path: str
    size: int
    group: str


@dataclass(frozen=True)
class DownloadProgress:
    downloaded: int
    total: int | None
    active: bool


def fetch_download_files(config: DeploymentConfig) -> tuple[DownloadFile, ...]:
    """Read sizes once, before rental; missing metadata must never block deploy."""
    repo = config.repo_id if config.backend == BackendType.LLAMACPP else config.model_name
    if not repo:
        return ()
    try:
        info = HfApi().model_info(repo, revision=config.revision or None, files_metadata=True, timeout=10)
        root = "models" if config.backend == BackendType.LLAMACPP else "hf/hub"
        cache = f"{root}/models--{repo.replace('/', '--')}/blobs"
        files: list[DownloadFile] = []
        incomplete: set[str] = set()
        for sibling in info.siblings or ():
            name, size = sibling.rfilename, sibling.size
            split = _SPLIT_GGUF.fullmatch(name)
            if name.endswith(".gguf"):
                group = split[1] if split else name
            elif name.endswith((".safetensors", ".bin")):
                path = PurePosixPath(name)
                group = str(path.parent / ("weights" + path.suffix))
            else:
                continue
            oid = sibling.lfs.sha256 if sibling.lfs else sibling.blob_id
            if not isinstance(size, int) or size <= 0 or not oid or not _BLOB.fullmatch(oid):
                incomplete.add(group)
                continue
            files.append(DownloadFile(f"{cache}/{oid}", size, group))
        # A missing shard size must not turn a partial manifest into 100%.
        for sibling in info.siblings or ():
            split = _SPLIT_GGUF.fullmatch(sibling.rfilename)
            if split and sum(file.group == split[1] for file in files) != int(split[3]):
                incomplete.add(split[1])
        return tuple(file for file in files if file.group not in incomplete)
    except Exception:
        return ()


def measure_download(output: str, files: tuple[DownloadFile, ...]) -> DownloadProgress | None:
    """Match observed blobs to whole shard groups, counting each blob once.

    Groups are selected by files actually on disk, not a second implementation
    of the runtime's quant selection. Snapshot symlinks are never counted.
    """
    observed: dict[str, tuple[int, bool]] = {}
    for line in output.splitlines():
        if not line.startswith("FILE "):
            continue
        try:
            size_text, blocks_text, path = line[5:].split(" ", 2)
            size, blocks = int(size_text), int(blocks_text)
            if size < 0 or blocks < 0:
                continue
        except ValueError:
            continue
        partial = path.endswith(_PARTIAL_SUFFIXES)
        if partial:
            path = path.rsplit(".", 1)[0]
            # Xet can preallocate sparse files: logical length is not progress.
            size = min(size, blocks * 512)
        previous = observed.get(path)
        if previous is None or not partial:
            observed[path] = (size, partial)
    if not observed:
        return None
    known = {file.path: file for file in files}
    groups = {known[path].group for path in observed if path in known}
    selected = {file.path: file for file in files if file.group in groups}
    unknown_active = any(partial and path not in known for path, (_, partial) in observed.items())
    if not selected or unknown_active:
        return DownloadProgress(
            sum(size for size, _ in observed.values()), None,
            any(partial for _, partial in observed.values()),
        )
    total = sum(file.size for file in selected.values())
    downloaded = sum(min(observed.get(path, (0, False))[0], file.size) for path, file in selected.items())
    active = downloaded < total or any(observed.get(path, (0, False))[1] for path in selected)
    return DownloadProgress(downloaded, total, active)
