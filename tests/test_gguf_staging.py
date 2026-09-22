"""Accelerated GGUF staging shared by Prime and Vast llama.cpp runtimes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from llm_launchpad.core.serving_runtime import (
    _GGUF_STAGE_PROGRAM,
    GGUF_FILE_PLACEHOLDER,
    gguf_exec_command,
    gguf_stage_args,
    gguf_stage_command,
)


class GgufStageCommandTests(unittest.TestCase):
    def test_staging_uses_xet_high_performance_and_pins_revision(self) -> None:
        setup, flag = gguf_stage_command(
            repo_id="acme/model-GGUF",
            revision="abc123",
            quant="Q4_K_M",
            dest_dir="/data/staged",
        )

        self.assertEqual(flag, ("--hf-file", "/data/staged/weights.args"))
        self.assertIn("HF_XET_HIGH_PERFORMANCE=1", setup)
        self.assertIn("snapshot_download", setup)
        self.assertIn("LLM_LAUNCHPAD_GGUF_REVISION=abc123", setup)
        self.assertIn("weights.args", setup)

    def test_staging_rejects_missing_repo_or_quant(self) -> None:
        with self.assertRaises(ValueError):
            gguf_stage_command(
                repo_id="not-a-repo", revision=None, quant="Q4_K_M", dest_dir="/tmp/x"
            )
        with self.assertRaises(ValueError):
            gguf_stage_command(
                repo_id="acme/model", revision=None, quant=" ", dest_dir="/tmp/x"
            )

    def test_stage_program_resolves_split_shards_and_emits_first_shard(self) -> None:
        self.assertIn("snapshot_download", _GGUF_STAGE_PROGRAM)
        self.assertIn("weights.args", _GGUF_STAGE_PROGRAM)
        self.assertIn("weights.list", _GGUF_STAGE_PROGRAM)
        # Every shard is staged; the server resolves siblings from the first.
        self.assertIn("allow_patterns", _GGUF_STAGE_PROGRAM)

    def test_stage_args_keep_repo_listing_with_pinned_file(self) -> None:
        args, setup = gguf_stage_args(
            repo_id="acme/model-GGUF",
            revision=None,
            quant="Q4_K_M",
            dest_dir="/data/staged",
        )

        self.assertEqual(args[:4], ["/app/llama-server", "--hf-repo", "acme/model-GGUF", "--hf-file"])
        self.assertIn("/data/staged/weights.args", args)
        self.assertIn("snapshot_download", setup)

    def test_the_prelude_runs_as_a_shell_script_and_reaches_the_downloader(self) -> None:
        """Run the real prelude in a real shell rather than grepping it.

        Substring assertions cannot see statement boundaries. The exports were
        once joined on a bare space, which reads as one ``export`` command
        taking the next ``export``, then ``python3`` and its ``-``, as variable
        names: every Prime and Vast llama.cpp rental died on
        "export: -: bad variable name" under ``set -eu`` before staging a byte,
        while every assertion in this file still passed.
        """

        shell = shutil.which("sh")
        if shell is None:  # pragma: no cover - POSIX hosts always have one
            self.skipTest("no POSIX shell available")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dest = root / "staged"
            captured = root / "captured.json"
            stub_dir = root / "bin"
            stub_dir.mkdir()
            stub = stub_dir / "python3"
            # Stands in for the staging interpreter and records what the shell
            # actually handed it: its argv, and the download environment the
            # exports were supposed to place there.
            stub.write_text(
                "#!/bin/sh\n"
                f"{shutil.which('python3')} -c \"import json,os,sys;"
                f"json.dump({{'argv': sys.argv[1:], 'env': dict(os.environ)}}, open({str(captured)!r}, 'w'))\""
                ' "$@"\n',
                encoding="utf-8",
            )
            stub.chmod(0o755)

            setup, _ = gguf_stage_command(
                repo_id="acme/model-GGUF", revision=None, quant="Q4_K_M",
                dest_dir=str(dest),
            )
            completed = subprocess.run(
                [shell, "-c", f"set -eu\n{setup}\ntrue"],
                env={**os.environ, "PATH": f"{stub_dir}{os.pathsep}{os.environ['PATH']}"},
                capture_output=True, text=True, timeout=60,
            )

            self.assertEqual(
                completed.returncode, 0,
                f"prelude failed: {completed.stderr.strip()!r}",
            )
            self.assertEqual(completed.stderr.strip(), "")
            recorded = json.loads(captured.read_text(encoding="utf-8"))
            # The program is handed to -c, then the repo, quant, destination
            # and cache the exports named, in that order. `python3 -` would
            # read the program from stdin instead, run nothing under
            # `nohup … < /dev/null`, and exit 0 having staged no weights.
            self.assertEqual(recorded["argv"][0], "-c")
            self.assertIn("snapshot_download", recorded["argv"][1])
            self.assertEqual(
                recorded["argv"][2:6],
                ["acme/model-GGUF", "Q4_K_M", str(dest), f"{dest}/hf-hub"],
            )
            self.assertEqual(recorded["env"]["HF_HUB_CACHE"], f"{dest}/hf-hub")
            self.assertEqual(recorded["env"]["HF_HUB_DOWNLOAD_TIMEOUT"], "120")
            self.assertEqual(recorded["env"]["HF_HUB_ETAG_TIMEOUT"], "30")

    def test_the_staging_program_actually_runs_and_writes_the_file_name(self) -> None:
        """Run the prelude against a real interpreter with the Hub stubbed.

        Every other test here checks the *text* of the command. The invocation
        was ``python3 -``, which reads the program from standard input -- empty
        under ``nohup … < /dev/null`` -- so the interpreter ran nothing, exited
        0, and left the program in argv. Staging wrote no ``weights.args`` on
        any Prime or Vast rental, and llama.cpp reported ``failed to load model
        ''`` about a downloader that had never run.
        """

        shell = shutil.which("sh")
        if shell is None:  # pragma: no cover - POSIX hosts always have one
            self.skipTest("no POSIX shell available")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dest = root / "staged"
            snapshot = root / "snapshot"
            (snapshot / "UD-Q2_K_XL").mkdir(parents=True)
            shard = snapshot / "UD-Q2_K_XL" / "Model-UD-Q2_K_XL-00001-of-00002.gguf"
            shard.write_text("x", encoding="utf-8")

            # Stands in for the Hub: the staging program imports these names at
            # runtime, so a stub package on PYTHONPATH is enough to run it for
            # real without a network round trip.
            stub_pkg = root / "stubs"
            (stub_pkg / "huggingface_hub").mkdir(parents=True)
            (stub_pkg / "huggingface_hub" / "__init__.py").write_text(
                "class _Sibling:\n"
                "    def __init__(self, name):\n"
                "        self.rfilename = name\n"
                "class _Info:\n"
                f"    siblings = [_Sibling('UD-Q2_K_XL/{shard.name}')]\n"
                "    sha = 'deadbeef'\n"
                "class HfApi:\n"
                "    def model_info(self, *a, **k):\n"
                "        return _Info()\n"
                f"def snapshot_download(**kwargs):\n"
                f"    return {str(snapshot)!r}\n",
                encoding="utf-8",
            )
            # Reproduce the runtime images: no Xet accelerator, and a Python
            # whose pip cannot install one. Shadowing both on PYTHONPATH keeps
            # the test offline and deterministic.
            (stub_pkg / "hf_xet.py").write_text(
                "raise ImportError('no xet in this image')", encoding="utf-8"
            )
            (stub_pkg / "pip").mkdir()
            (stub_pkg / "pip" / "__init__.py").write_text(
                "raise SystemExit('pip is not usable in this image')", encoding="utf-8"
            )

            setup, flag = gguf_stage_command(
                repo_id="acme/model-GGUF", revision=None, quant="UD-Q2_K_XL",
                dest_dir=str(dest),
            )
            completed = subprocess.run(
                [shell, "-c", f"set -eu\n{setup}\ntrue"],
                env={**os.environ, "PYTHONPATH": str(stub_pkg)},
                capture_output=True, text=True, timeout=120,
                stdin=subprocess.DEVNULL,
            )

            self.assertEqual(
                completed.returncode, 0,
                f"staging failed: {completed.stderr.strip()[-600:]!r}",
            )
            self.assertIn("staged 1 GGUF file(s)", completed.stdout)
            # hf-xet could not be installed here either, and that is survivable:
            # making it mandatory aborted every rental on an image whose Python
            # has no working pip.
            self.assertIn("staging without the Xet accelerator", completed.stdout)
            written = Path(flag[1])
            self.assertTrue(written.is_file(), "staging wrote no weights.args")
            self.assertEqual(
                written.read_text(encoding="utf-8").strip(),
                f"UD-Q2_K_XL/{shard.name}",
            )

    def test_the_staged_file_name_reaches_the_server_as_one_argument(self) -> None:
        """Run the exec line in a shell with llama-server stubbed out.

        ``shlex.join`` quotes every argument, so the ``$(cat …)`` both providers
        wrote into their argument lists was passed through as that literal text.
        llama.cpp resolved the nonsense against the repo listing and reported
        ``failed to load model ''``, which reads as a bad model rather than as a
        bad command line -- and cost a rental per diagnosis.
        """

        shell = shutil.which("sh")
        if shell is None:  # pragma: no cover - POSIX hosts always have one
            self.skipTest("no POSIX shell available")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "weights.args"
            # A name with a space is the case a shell splits if the variable is
            # referenced unquoted.
            weights.write_text("UD-Q2_K_XL/Model One-00001-of-00004.gguf\n", encoding="utf-8")
            captured = root / "argv.json"
            stub_dir = root / "app"
            stub_dir.mkdir()
            stub = stub_dir / "llama-server"
            stub.write_text(
                "#!/bin/sh\n"
                f"{shutil.which('python3')} -c \"import json,sys;"
                f"json.dump(sys.argv[1:], open({str(captured)!r}, 'w'))\""
                ' "$@"\n',
                encoding="utf-8",
            )
            stub.chmod(0o755)

            command = gguf_exec_command(
                [
                    str(stub), "--hf-repo", "acme/model-GGUF:UD-Q2_K_XL",
                    "--hf-file", GGUF_FILE_PLACEHOLDER, "--ctx-size", "262144",
                ],
                weights_args_path=str(weights),
            )
            completed = subprocess.run(
                [shell, "-c", f"set -eu\n{command}"],
                capture_output=True, text=True, timeout=60,
            )

            self.assertEqual(
                completed.returncode, 0, f"exec line failed: {completed.stderr.strip()!r}"
            )
            argv = json.loads(captured.read_text(encoding="utf-8"))
            self.assertEqual(
                argv,
                [
                    "--hf-repo", "acme/model-GGUF:UD-Q2_K_XL",
                    "--hf-file", "UD-Q2_K_XL/Model One-00001-of-00004.gguf",
                    "--ctx-size", "262144",
                ],
            )

    def test_without_a_staged_name_the_server_resolves_the_quant_itself(self) -> None:
        """The pinned llama.cpp image cannot stage, and must still serve.

        It carries no huggingface_hub, no hf-xet and no pip, so the accelerated
        staging path can never run there. Treating that as a failed deploy left
        every Vast rental dying before it served anything; llama.cpp's own
        ``--hf-repo`` download -- the path this provider was certified on
        before staging existed -- is the fallback, and the now-meaningless
        ``--hf-file`` flag has to go with its value rather than swallowing the
        next argument.
        """

        shell = shutil.which("sh")
        if shell is None:  # pragma: no cover - POSIX hosts always have one
            self.skipTest("no POSIX shell available")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            captured = root / "argv.json"
            stub = root / "llama-server"
            stub.write_text(
                "#!/bin/sh\n"
                f"{shutil.which('python3')} -c \"import json,sys;"
                f"json.dump(sys.argv[1:], open({str(captured)!r}, 'w'))\""
                ' "$@"\n',
                encoding="utf-8",
            )
            stub.chmod(0o755)

            command = gguf_exec_command(
                [
                    str(stub), "--hf-repo", "acme/model-GGUF:Q4_K_M",
                    "--hf-file", GGUF_FILE_PLACEHOLDER, "--ctx-size", "4096",
                ],
                weights_args_path=str(root / "absent.args"),
            )
            completed = subprocess.run(
                [shell, "-c", f"set -eu\n{command}"],
                capture_output=True, text=True, timeout=60,
            )

            self.assertEqual(
                completed.returncode, 0, f"fallback failed: {completed.stderr.strip()!r}"
            )
            self.assertEqual(
                json.loads(captured.read_text(encoding="utf-8")),
                ["--hf-repo", "acme/model-GGUF:Q4_K_M", "--ctx-size", "4096"],
            )
            self.assertIn("letting llama.cpp resolve the quant", completed.stderr)

    def test_explicit_xet_disable_is_respected(self) -> None:
        with patch.dict("os.environ", {"HF_HUB_DISABLE_XET": "1"}, clear=False):
            setup, _ = gguf_stage_command(
                repo_id="acme/model-GGUF",
                revision=None,
                quant="Q4_K_M",
                dest_dir="/data/staged",
            )
        self.assertIn("HF_HUB_DISABLE_XET=1", setup)
        self.assertNotIn("HF_XET_HIGH_PERFORMANCE", setup)


if __name__ == "__main__":
    unittest.main()
