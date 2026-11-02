"""Require a release tag to name the package version on the main branch."""
from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess


def check_release(root: Path, tag: str) -> None:
    """Reject mismatched versions and tagged commits outside reviewed main."""
    module = ast.parse((root / "llm_launchpad/_version.py").read_text())
    version = next(
        (
            node.value.value
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ),
        None,
    )
    if not version or tag != f"v{version}":
        raise ValueError("Release tag must match the package version in llm_launchpad/_version.py.")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", "HEAD", "refs/remotes/origin/main"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError("Release commit must belong to origin/main; fetch main before validation.")


if __name__ == "__main__":
    check_release(Path(__file__).resolve().parents[1], os.environ.get("RELEASE_TAG", ""))
