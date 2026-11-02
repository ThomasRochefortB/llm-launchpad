from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest

from scripts.check_release import check_release


class ReleaseSecurityTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.git("init", "--initial-branch=main")
        self.git("config", "user.name", "Release test")
        self.git("config", "user.email", "release@example.invalid")
        package = self.root / "llm_launchpad"
        package.mkdir()
        (package / "_version.py").write_text('__version__ = "1.2.3"\n')
        self.git("add", ".")
        self.git("commit", "-m", "Reviewed main commit")
        self.git("update-ref", "refs/remotes/origin/main", "HEAD")

    def git(self, *args: str) -> None:
        subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True)

    def test_version_on_main_can_be_released(self) -> None:
        check_release(self.root, "v1.2.3")

    def test_mismatched_version_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "package version"):
            check_release(self.root, "v1.2.4")

    def test_unreviewed_branch_cannot_be_released(self) -> None:
        self.git("checkout", "-b", "unreviewed")
        (self.root / "unreviewed.txt").write_text("unreviewed change")
        self.git("add", ".")
        self.git("commit", "-m", "Unreviewed commit")
        with self.assertRaisesRegex(ValueError, "origin/main"):
            check_release(self.root, "v1.2.3")

    def test_ancestor_of_main_can_be_released(self) -> None:
        self.git("branch", "release-commit")
        (self.root / "later.txt").write_text("Later reviewed work")
        self.git("add", ".")
        self.git("commit", "-m", "Later main commit")
        self.git("update-ref", "refs/remotes/origin/main", "HEAD")
        self.git("checkout", "release-commit")
        check_release(self.root, "v1.2.3")
