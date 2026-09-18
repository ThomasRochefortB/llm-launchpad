"""Every provider must start llama.cpp with the metrics endpoint enabled.

The fleet reads tokens and throughput from the runtime's own /metrics, so a
launch that omits ``--metrics`` leaves that endpoint permanently blank.
"""

from __future__ import annotations

import unittest

from dataclasses import replace
from unittest.mock import patch

from llm_launchpad.core.prime_backend import PrimeBackend, resolve_prime_launch_spec
from llm_launchpad.core.vast_runtime import vast_runtime_script
from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.models import (
    DeploymentConfig,
    PrimeProviderOptions,
    VastProviderOptions,
)


def _llamacpp_config(provider: ComputeProvider, **overrides: object) -> DeploymentConfig:
    provider_options = (
        VastProviderOptions("1001", 100, 0.42, "42")
        if provider == ComputeProvider.VAST
        else PrimeProviderOptions(disk_id="disk-1")
    )
    base = DeploymentConfig(
        backend=BackendType.LLAMACPP,
        provider=provider,
        app_name="llp-llamacpp-metrics",
        repo_id="acme/model-GGUF",
        quant="Q4_K_M",
        gpu_type="RTX 4090",
        gpu_count=1,
        do_deploy=True,
        endpoint_api_key="endpoint-secret",
        gguf_architecture="llama",
        provider_options=provider_options,
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


class LlamaCppMetricsFlagTests(unittest.TestCase):
    def _server_line(self, script: str) -> str:
        # The staging python program mentions llama-server in a comment; the
        # actual server invocation is the exec line.
        candidates = [
            line for line in script.splitlines() if line.startswith("exec /app/llama-server")
        ]
        self.assertTrue(candidates, "expected an exec llama-server line in the startup script")
        return candidates[0]

    def test_vast_starts_llama_server_with_metrics(self) -> None:
        script = vast_runtime_script(_llamacpp_config(ComputeProvider.VAST))
        server_line = self._server_line(script)
        self.assertIn("--metrics", server_line)

    def test_prime_starts_llama_server_with_metrics(self) -> None:
        config = _llamacpp_config(ComputeProvider.PRIME)
        command = PrimeBackend._bootstrap_docker_command(config, resolve_prime_launch_spec(config))
        self.assertIn("--metrics", " ".join(command))

    def test_user_server_args_do_not_displace_the_flag(self) -> None:
        config = _llamacpp_config(ComputeProvider.VAST, server_args="--ctx-size 65536")
        script = vast_runtime_script(config)
        server_line = self._server_line(script)
        self.assertIn("--ctx-size 65536", server_line)
        self.assertIn("--metrics", server_line)

    def test_modal_still_starts_llama_server_with_metrics(self) -> None:
        from llm_launchpad.backends import modal_llamacpp_app

        config = {
            "repo_id": "unsloth/Test-GGUF",
            "quant": "Q4_K_M",
            "revision": None,
            "served_model_name": "test-model",
            "server_args": ["--ctx-size", "131072"],
            "host": "0.0.0.0",
            "port": 8080,
            "n_gpu_layers": None,
        }
        with (
            patch.object(modal_llamacpp_app, "_load_config", return_value=config),
            patch.object(
                modal_llamacpp_app,
                "_resolve_or_download_model_entrypoint",
                return_value="/models/test.gguf",
            ),
            patch.object(
                modal_llamacpp_app,
                "_resolve_llama_server_binary",
                return_value="llama-server",
            ),
            patch.object(
                modal_llamacpp_app, "_llama_server_runtime_env", return_value=({}, None)
            ),
            patch.object(modal_llamacpp_app, "_warm_volume_paths"),
            patch.object(modal_llamacpp_app.subprocess, "Popen") as popen,
        ):
            modal_llamacpp_app.serve.local()

        self.assertIn("--metrics", list(popen.call_args.args[0]))


if __name__ == "__main__":
    unittest.main()
