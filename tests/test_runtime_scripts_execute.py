"""Run the whole generated Prime and Vast startup scripts in a real shell.

Every defect 623ba8b shipped lived in a generated shell script, and every one
passed the suite: substring assertions cannot see statement boundaries,
argument order, quoting, or which stream an interpreter reads its program
from. ``test_gguf_staging`` executes the staging helpers one at a time; this
module executes each provider's *complete* script end to end -- prelude,
staging, fit guard and exec line -- with the binaries a rental would run
replaced by stubs that record the argv and environment they were handed.

Only absolute paths are remapped into a scratch directory. The shell text,
the staging interpreter and the quoting are the ones a real host runs.
"""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from typing import Any
import unittest
from unittest.mock import patch

from llm_launchpad.core.prime_backend import (
    PRIME_RUNTIME_ROOT,
    PrimeBackend,
    resolve_prime_launch_spec,
)
from llm_launchpad.core.vast_runtime import VAST_RUNTIME_DIR, vast_runtime_script
from llm_launchpad.protocol.enums import (
    BackendType,
    CertificationState,
    ComputeProvider,
    ServingObjective,
)
from llm_launchpad.protocol.models import (
    DeploymentConfig,
    MemoryEstimate,
    PlacementAssessment,
    PrimeProviderOptions,
    RuntimeTuning,
    ServingRequirements,
    VastProviderOptions,
)

ENDPOINT_KEY = "endpoint-secret"
REPO = "acme/model-GGUF"
QUANT = "UD-Q2_K_XL"
# A space is the case a shell splits when the variable is referenced unquoted.
SHARD = f"{QUANT}/Model One-00001-of-00002.gguf"


def _vast_llamacpp(**overrides: Any) -> DeploymentConfig:
    base = DeploymentConfig(
        backend=BackendType.LLAMACPP,
        provider=ComputeProvider.VAST,
        app_name="llp-vast-llamacpp-test",
        repo_id=REPO,
        quant=QUANT,
        gpu_type="RTX 4090",
        gpu_count=1,
        do_deploy=True,
        provider_options=VastProviderOptions("1001", 100, 0.42, "42"),
        gguf_architecture="llama",
        endpoint_api_key=ENDPOINT_KEY,
        served_model_name="acme-model",
    )
    return replace(base, **overrides)


def _vast_vllm(**overrides: Any) -> DeploymentConfig:
    return _vast_llamacpp(
        backend=BackendType.VLLM, model_name="acme/model", n_gpu=1, **overrides
    )


def _prime_llamacpp() -> DeploymentConfig:
    tuning = RuntimeTuning(
        parallel_slots=4, batch_size=2048, ubatch_size=64,
        cache_type_k="f16", cache_type_v="f16", flash_attention=False,
        gpu_layers="all", fit_target_mib=2048,
    )
    return DeploymentConfig(
        backend=BackendType.LLAMACPP,
        provider=ComputeProvider.PRIME,
        repo_id=REPO,
        quant=QUANT,
        served_model_name="acme-model",
        endpoint_api_key=ENDPOINT_KEY,
        gpu_type="H100",
        provider_options=PrimeProviderOptions(),
        runtime_tuning=tuning,
        serving_requirements=ServingRequirements(
            context_tokens=32768, objective=ServingObjective.GENERAL_PURPOSE,
            full_context_per_request=True, gpu_only=True,
        ),
        # Present so the fit guard is part of the script under test.
        placement_assessment=PlacementAssessment(
            fits=True, gpu_resident=True, tuning=tuning,
            certification=CertificationState.ESTIMATED,
            fingerprint="fingerprint-under-test",
            memory=MemoryEstimate(
                weights_gb=20.0, kv_cache_gb=8.0, compute_gb=2.0,
                attention_scratch_gb=1.0, speculative_gb=0.0,
                reserve_gb=0.0, total_gb=31.0,
                per_device_required_gb=(31.0,), confidence=0.82,
                source="gguf-metadata", total_layer_count=32,
            ),
        ),
    )


def _prime_vllm(**overrides: Any) -> DeploymentConfig:
    base = DeploymentConfig(
        backend=BackendType.VLLM,
        provider=ComputeProvider.PRIME,
        model_name="acme/model",
        served_model_name="acme-model",
        endpoint_api_key=ENDPOINT_KEY,
        provider_options=PrimeProviderOptions(),
    )
    return replace(base, **overrides)


class _Host:
    """A scratch directory standing in for the rented machine."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.bin = root / "bin"
        self.bin.mkdir()
        self.records = root / "records"
        self.records.mkdir()
        self.stubs = root / "pystubs"
        self.stubs.mkdir()
        self._python = shutil.which("python3")
        if self._python is None:  # pragma: no cover - the suite runs on python3
            raise unittest.SkipTest("no python3 on PATH")

    def recorder(self, name: str, *, exit_code: int = 0) -> Path:
        """Install ``name`` on PATH as a stub that appends its argv and env."""

        log = self.records / f"{name}.jsonl"
        stub = self.bin / name
        stub.write_text(
            "#!/bin/sh\n"
            f"{self._python} -c \"import json,os,sys;"
            f"open({str(log)!r}, 'a').write(json.dumps({{'argv': sys.argv[1:], "
            "'env': dict(os.environ)})+chr(10))\""
            f' "$@"\nexit {exit_code}\n',
            encoding="utf-8",
        )
        stub.chmod(0o755)
        return stub

    def calls(self, name: str) -> list[dict[str, Any]]:
        log = self.records / f"{name}.jsonl"
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]

    def hub_serving(self, shard: str) -> None:
        """Stand in for the Hub with a repo that lists ``shard``."""

        snapshot = self.root / "snapshot"
        (snapshot / Path(shard).parent).mkdir(parents=True)
        (snapshot / shard).write_text("x", encoding="utf-8")
        package = self.stubs / "huggingface_hub"
        package.mkdir()
        (package / "__init__.py").write_text(
            "class _Sibling:\n"
            "    def __init__(self, name):\n"
            "        self.rfilename = name\n"
            "class _Info:\n"
            f"    siblings = [_Sibling({shard!r})]\n"
            "    sha = 'deadbeef'\n"
            "class HfApi:\n"
            "    def model_info(self, *a, **k):\n"
            "        return _Info()\n"
            "def snapshot_download(**kwargs):\n"
            f"    return {str(snapshot)!r}\n",
            encoding="utf-8",
        )
        self._image_without_pip()

    def hub_missing(self) -> None:
        """Reproduce the pinned llama.cpp image: no huggingface_hub at all."""

        package = self.stubs / "huggingface_hub"
        package.mkdir()
        (package / "__init__.py").write_text(
            "raise ImportError('No module named huggingface_hub')", encoding="utf-8"
        )
        self._image_without_pip()

    def _image_without_pip(self) -> None:
        # Offline and deterministic: the staging program tries to pip-install
        # hf-xet, which must be survivable on an image with no usable pip.
        (self.stubs / "hf_xet.py").write_text(
            "raise ImportError('no xet in this image')", encoding="utf-8"
        )
        (self.stubs / "pip").mkdir()
        (self.stubs / "pip" / "__init__.py").write_text(
            "raise SystemExit('pip is not usable in this image')", encoding="utf-8"
        )

    def run(
        self, shell: str, script: str, *, env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        executable = shutil.which(shell)
        if executable is None:  # pragma: no cover - POSIX hosts always have sh
            raise unittest.SkipTest(f"no {shell} available")
        return subprocess.run(
            [executable, "-c", script],
            env={
                "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
                "PYTHONPATH": str(self.stubs),
                "HOME": str(self.root),
                **(env or {}),
            },
            # A detached rental runs under nohup with stdin closed; a program
            # that reads its code from stdin must fail here the way it did there.
            stdin=subprocess.DEVNULL,
            capture_output=True, text=True, timeout=120,
        )


def _flag(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def _env_file(text: str) -> dict[str, str]:
    """Parse a Docker ``--env-file``: KEY=VALUE per line, no quoting."""

    values: dict[str, str] = {}
    for line in text.splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            values[key] = value
    return values


class VastScriptExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.host = _Host(Path(directory.name))
        patcher = patch("llm_launchpad.core.vast_runtime.get_token", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _localise(self, script: str) -> str:
        return script.replace(VAST_RUNTIME_DIR, str(self.host.root / "runtime")).replace(
            "/app/", f"{self.host.bin}/"
        )

    def test_llamacpp_script_stages_and_starts_the_server_on_the_staged_shard(self) -> None:
        self.host.hub_serving(SHARD)
        self.host.recorder("llama-server")

        completed = self.host.run("sh", self._localise(vast_runtime_script(_vast_llamacpp())))

        self.assertEqual(completed.returncode, 0, completed.stderr[-800:])
        self.assertIn("llm-launchpad runtime starting", completed.stdout)
        self.assertIn("staged 1 GGUF file(s)", completed.stdout)
        (call,) = self.host.calls("llama-server")
        argv = call["argv"]
        self.assertEqual(_flag(argv, "--hf-repo"), f"{REPO}:{QUANT}")
        self.assertEqual(_flag(argv, "--hf-file"), SHARD)
        self.assertEqual(_flag(argv, "--alias"), "acme-model")
        self.assertIn("--metrics", argv)
        self.assertIn("--no-mmproj", argv)
        # The key travels by environment; argv is readable by anything that
        # can list processes on the host.
        self.assertEqual(call["env"]["LLAMA_API_KEY"], ENDPOINT_KEY)
        self.assertNotIn(ENDPOINT_KEY, argv)

    def test_llamacpp_script_falls_back_to_llamacpp_download_without_huggingface_hub(
        self,
    ) -> None:
        self.host.hub_missing()
        self.host.recorder("llama-server")

        completed = self.host.run("sh", self._localise(vast_runtime_script(_vast_llamacpp())))

        self.assertEqual(completed.returncode, 0, completed.stderr[-800:])
        (call,) = self.host.calls("llama-server")
        argv = call["argv"]
        # No dangling --hf-file for llama.cpp to read the next flag as.
        self.assertNotIn("--hf-file", argv)
        self.assertEqual(_flag(argv, "--hf-repo"), f"{REPO}:{QUANT}")
        self.assertEqual(_flag(argv, "--host"), "127.0.0.1")

    def test_vllm_script_serves_with_stated_limits_and_the_key_in_the_environment(
        self,
    ) -> None:
        self.host.recorder("vllm")
        config = _vast_vllm(max_context_tokens=32768)

        completed = self.host.run("sh", self._localise(vast_runtime_script(config)))

        self.assertEqual(completed.returncode, 0, completed.stderr[-800:])
        (call,) = self.host.calls("vllm")
        argv = call["argv"]
        self.assertEqual(argv[:2], ["serve", "acme/model"])
        self.assertEqual(_flag(argv, "--max-model-len"), "32768")
        self.assertEqual(_flag(argv, "--max-num-seqs"), "256")
        self.assertEqual(_flag(argv, "--served-model-name"), "acme-model")
        self.assertEqual(call["env"]["VLLM_API_KEY"], ENDPOINT_KEY)
        self.assertNotIn(ENDPOINT_KEY, argv)


class PrimeScriptExecutionTests(unittest.TestCase):
    """The outer bootstrap, then the container command docker was handed."""

    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.host = _Host(Path(directory.name))
        self.runtime_root = self.host.root / "opt"
        self.runtime_root.mkdir()
        # Source-build recipes redirect this into `docker build`.
        (self.runtime_root / "runtime.dockerfile").write_text("", encoding="utf-8")
        patcher = patch("huggingface_hub.get_token", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _bootstrap(self, config: DeploymentConfig) -> tuple[list[str], dict[str, str]]:
        """Run the bootstrap script; return the `docker run` argv and its env file."""

        self.host.recorder("docker")
        self.host.recorder("nvidia-smi")
        launch = resolve_prime_launch_spec(config)
        script = PrimeBackend._bootstrap_script(config, launch).replace(
            PRIME_RUNTIME_ROOT, str(self.runtime_root)
        )

        completed = self.host.run("bash", script)

        self.assertEqual(completed.returncode, 0, completed.stderr[-800:])
        self.assertEqual(
            (self.runtime_root / "bootstrap.exit").read_text(encoding="utf-8").strip(), "0"
        )
        runs = [call["argv"] for call in self.host.calls("docker") if call["argv"][:1] == ["run"]]
        self.assertEqual(len(runs), 1, self.host.calls("docker"))
        run = runs[0]
        self.assertEqual(_flag(run, "--env-file"), f"{self.runtime_root}/runtime.env")
        return run, _env_file(PrimeBackend._docker_env_file(config))

    def _container(
        self, run: list[str], env: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        """Run the container command exactly as docker would hand it to the shell."""

        entrypoint = _flag(run, "--entrypoint")
        inner = _flag(run, "-lc")
        for container_path in ("/root/.cache/llama.cpp", "/data/llama.cpp"):
            inner = inner.replace(container_path, str(self.host.root / "cache"))
        inner = inner.replace("/app/", f"{self.host.bin}/")
        return self.host.run(Path(entrypoint).name, inner, env=env)

    def test_llamacpp_container_stages_fits_and_serves_the_staged_shard(self) -> None:
        self.host.hub_serving(SHARD)
        self.host.recorder("llama-fit-params")
        self.host.recorder("llama-server")
        run, env = self._bootstrap(_prime_llamacpp())
        # The endpoint key is in the env file, never in the uploaded script.
        self.assertNotIn(ENDPOINT_KEY, run)
        self.assertEqual(env["LLAMA_ARG_API_KEY"], ENDPOINT_KEY)

        completed = self._container(run, env)

        self.assertEqual(completed.returncode, 0, completed.stderr[-800:])
        (fit,) = self.host.calls("llama-fit-params")
        self.assertEqual(_flag(fit["argv"], "--hf-file"), SHARD)
        self.assertIn("LLM_LAUNCHPAD_ATTESTATION_JSON_BEGIN", completed.stdout)
        (server,) = self.host.calls("llama-server")
        argv = server["argv"]
        self.assertEqual(_flag(argv, "--hf-repo"), f"{REPO}:{QUANT}")
        self.assertEqual(_flag(argv, "--hf-file"), SHARD)
        self.assertEqual(_flag(argv, "--api-key"), ENDPOINT_KEY)
        self.assertEqual(_flag(argv, "--alias"), "acme-model")
        self.assertIn("--metrics", argv)

    def test_llamacpp_container_refuses_to_serve_a_plan_the_planner_rejects(self) -> None:
        self.host.hub_serving(SHARD)
        self.host.recorder("llama-fit-params", exit_code=3)
        self.host.recorder("llama-server")
        run, env = self._bootstrap(_prime_llamacpp())

        completed = self._container(run, env)

        self.assertEqual(completed.returncode, 3)
        self.assertEqual(self.host.calls("llama-server"), [])

    def test_llamacpp_container_falls_back_without_huggingface_hub(self) -> None:
        self.host.hub_missing()
        self.host.recorder("llama-fit-params")
        self.host.recorder("llama-server")
        run, env = self._bootstrap(_prime_llamacpp())

        completed = self._container(run, env)

        self.assertEqual(completed.returncode, 0, completed.stderr[-800:])
        # Nothing staged, so there is no shard to plan against; the server
        # resolves the quant itself.
        self.assertEqual(self.host.calls("llama-fit-params"), [])
        (server,) = self.host.calls("llama-server")
        self.assertNotIn("--hf-file", server["argv"])
        self.assertEqual(_flag(server["argv"], "--api-key"), ENDPOINT_KEY)

    def test_vllm_container_serves_with_stated_limits_and_the_endpoint_key(self) -> None:
        self.host.recorder("vllm")
        run, env = self._bootstrap(_prime_vllm(max_concurrent_sequences=16))
        self.assertNotIn(ENDPOINT_KEY, run)

        completed = self._container(run, env)

        self.assertEqual(completed.returncode, 0, completed.stderr[-800:])
        (call,) = self.host.calls("vllm")
        argv = call["argv"]
        self.assertEqual(argv[:2], ["serve", "acme/model"])
        self.assertEqual(_flag(argv, "--max-num-seqs"), "16")
        self.assertNotIn("--max-model-len", argv)
        self.assertEqual(_flag(argv, "--api-key"), ENDPOINT_KEY)


if __name__ == "__main__":
    unittest.main()
