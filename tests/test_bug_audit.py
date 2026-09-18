"""Reproductions for the first ten issues in the repository bug audit."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
from unittest.mock import patch

import pytest

from llm_launchpad.core import benchmark, connection_store, opencode
from llm_launchpad.core.coerce import optional_float
from llm_launchpad.protocol.enums import BackendType
from llm_launchpad.protocol.models import (
    BenchmarkConcurrencyResult,
    BenchmarkConfig,
    DeploymentConfig,
    EndpointInfo,
)


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), "1e9999", 10**400])
def test_scalar_parser_rejects_nonfinite_and_overflow(value: object) -> None:
    assert optional_float(value) is None


@pytest.mark.parametrize("changed", [{"app_id": "new"}, {"web_url": "https://new.example"}])
def test_replacement_deployment_does_not_inherit_cached_credentials(tmp_path: Path, changed: dict) -> None:
    path = tmp_path / "connections.json"
    path.write_text(json.dumps({"entries": {"vllm-demo": {
        "resource_id": "old", "base_url": "https://old.example/v1",
        "api_key": "old-secret", "model_id": "old-model", "max_context_tokens": 8192,
    }}}))
    row = EndpointInfo(name="vllm-demo", backend=BackendType.VLLM, **changed)
    connection_store.merge_connections([row], path, tmp_path / "missing.json")
    assert row.endpoint_api_key is None
    assert row.served_model_name is None
    assert row.max_context_tokens is None


def test_cached_trailing_slash_does_not_duplicate_api_version(tmp_path: Path) -> None:
    path = tmp_path / "connections.json"
    path.write_text(json.dumps({"entries": {"vllm-demo": {
        "backend": "vllm", "base_url": "https://demo.example/v1/",
    }}}))
    rows = connection_store.rows_from_connection_cache(path, tmp_path / "missing.json")
    assert rows[0].web_url == "https://demo.example"
    row = EndpointInfo(name="vllm-demo", backend=BackendType.VLLM)
    connection_store.merge_connections([row], path, tmp_path / "missing.json")
    assert row.web_url == "https://demo.example"


def test_failed_benchmark_is_excluded_from_best_result(tmp_path: Path) -> None:
    results = [
        BenchmarkConcurrencyResult(1, [], "", metrics={"output_token_throughput": 10}),
        BenchmarkConcurrencyResult(8, [], "", success=False, metrics={"output_token_throughput": 100}),
    ]
    summary = benchmark.build_run_summary(BenchmarkConfig(), tmp_path, results)
    assert summary.best_concurrency == 1
    assert summary.success is False


def test_empty_benchmark_sweep_is_not_successful(tmp_path: Path) -> None:
    assert benchmark.build_run_summary(BenchmarkConfig(), tmp_path, []).success is False


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_invalid_benchmark_metrics_do_not_suppress_csv_fallback(tmp_path: Path, value: str) -> None:
    json_path, csv_path = benchmark.expected_export_paths(tmp_path)
    json_path.write_text(json.dumps({"output_token_throughput": {"value": value}}))
    csv_path.write_text("Metric,avg\nOutput Token Throughput,12.5\n")
    metrics, source = benchmark.parse_aiperf_summary(json_path, csv_path)
    assert source == str(csv_path)
    assert metrics["output_token_throughput"] == 12.5


def test_connection_lookup_does_not_return_different_fallback_app() -> None:
    config = DeploymentConfig(backend=BackendType.VLLM, app_name="vllm-other")
    assert opencode.resolve_connection_for_app(
        "vllm-requested", rows=[], fallback_config=config,
        fallback_server_url="https://other.example",
    ) is None


def test_concurrent_atomic_writers_use_independent_temporary_files(tmp_path: Path) -> None:
    path = tmp_path / "opencode.json"
    barrier = threading.Barrier(2)
    original_replace = Path.replace

    def overlapping_replace(source: Path, target: Path) -> Path:
        barrier.wait(timeout=10)
        return original_replace(source, target)

    with patch.object(Path, "replace", overlapping_replace):
        with ThreadPoolExecutor(max_workers=2) as pool:
            writes = [pool.submit(opencode._atomic_write, path, text) for text in ('{"a": 1}', '{"b": 2}')]
            for write in writes:
                write.result()
    assert json.loads(path.read_text()) in ({"a": 1}, {"b": 2})
    assert list(tmp_path.iterdir()) == [path]


def test_recovery_write_preserves_last_valid_backup(tmp_path: Path) -> None:
    path = tmp_path / "opencode.json"
    backup = path.with_suffix(".json.bak")
    path.write_text('{"provider":')
    backup.write_text('{"provider": {}, "theme": "dark"}')
    recovered = opencode._load_opencode_config(path)
    recovered["model"] = "demo"
    opencode._write_opencode_config(path, recovered)
    assert json.loads(backup.read_text()) == {"provider": {}, "theme": "dark"}


@pytest.mark.parametrize("text", ['{"port": 4/* comment */096}', '{"provider": {}} /* unfinished'])
def test_malformed_jsonc_is_rejected_instead_of_silently_rewritten(text: str) -> None:
    with pytest.raises(ValueError):
        opencode._parse_jsonc(text)
