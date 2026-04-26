"""AIPerf benchmark command construction and result parsing."""

from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Iterable

from ..protocol.enums import BackendType
from ..protocol.models import (
    BenchmarkConcurrencyResult,
    BenchmarkConfig,
    BenchmarkRunSummary,
    EndpointInfo,
)
from .config import SETTINGS_DIR
from .naming import default_llamacpp_served_model_name, default_served_model_name, slugify_instance_name

DEFAULT_CONCURRENCY = [1, 2, 4, 8, 16]
DEFAULT_INPUT_TOKENS = 550
DEFAULT_OUTPUT_TOKENS = 256
DEFAULT_TOKENIZER = "gpt2"
DEFAULT_RANDOM_SEED = 42
DEFAULT_REQUEST_TIMEOUT_SECONDS = 300
BENCHMARKS_DIR = SETTINGS_DIR / "benchmarks"
SUMMARY_FILENAME = "benchmark_summary.json"

METRIC_KEYS = (
    "output_token_throughput",
    "request_throughput",
    "request_count",
    "time_to_first_token_avg",
    "time_to_first_token_p50",
    "time_to_first_token_p90",
    "time_to_first_token_p99",
    "inter_token_latency_avg",
    "inter_token_latency_p90",
    "request_latency_avg",
    "request_latency_p90",
    "request_latency_p99",
)

_METRIC_ALIASES = {
    "output_token_throughput": (
        "outputtokenthroughput",
        "outputtokenthroughputtokenssec",
        "outputtokenthroughputtokenspersec",
    ),
    "request_throughput": (
        "requestthroughput",
        "requestthroughputrequestssec",
        "requestthroughputrequestspersec",
    ),
    "request_count": ("requestcount", "requestcountrequests"),
    "time_to_first_token": (
        "timetofirsttoken",
        "ttft",
        "timetofirstoutputtoken",
    ),
    "inter_token_latency": ("intertokenlatency", "itl"),
    "request_latency": ("requestlatency", "latency"),
}


def parse_concurrency_values(value: str | Iterable[int] | None) -> list[int]:
    """Parse a comma/space separated concurrency list into positive integers."""
    if value is None:
        return list(DEFAULT_CONCURRENCY)
    if isinstance(value, str):
        parts = [part for part in re.split(r"[\s,]+", value.strip()) if part]
    else:
        parts = [str(item) for item in value]
    values: list[int] = []
    for part in parts:
        try:
            parsed = int(part)
        except ValueError as exc:
            raise ValueError(f"Invalid concurrency value: {part!r}") from exc
        if parsed < 1:
            raise ValueError("Concurrency values must be >= 1.")
        if parsed not in values:
            values.append(parsed)
    if not values:
        raise ValueError("At least one concurrency value is required.")
    return values


def request_count_for_concurrency(concurrency: int, override: int | None = None) -> int:
    """Return the default or overridden request count for a concurrency run."""
    if override is not None:
        if override < 1:
            raise ValueError("Request count must be >= 1.")
        return override
    return max(24, concurrency * 4)


def aiperf_cli_path() -> str | None:
    """Resolve the AIPerf CLI, preferring the active environment's scripts dir."""
    candidates = [
        Path(sys.prefix) / "bin" / "aiperf",
        Path(sys.prefix) / "Scripts" / "aiperf.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return shutil.which("aiperf")


def normalize_aiperf_url(server_url: str) -> str:
    """Return the endpoint root URL expected by AIPerf."""
    text = (server_url or "").strip().rstrip("/")
    if text.endswith("/v1"):
        return text[: -len("/v1")].rstrip("/")
    return text


def default_benchmark_run_dir(app_name: str, now: datetime | None = None) -> Path:
    """Build the default benchmark run directory for an app."""
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    safe_app_name = slugify_instance_name(app_name or "endpoint")
    return BENCHMARKS_DIR / safe_app_name / timestamp


def benchmark_config_from_endpoint(
    row: EndpointInfo | None,
    *,
    backend: BackendType,
    username: str = "",
    app_name: str | None = None,
    instance_name: str | None = None,
    server_url: str | None = None,
    model_name: str | None = None,
    concurrency: list[int] | None = None,
    request_count: int | None = None,
    input_tokens: int = DEFAULT_INPUT_TOKENS,
    output_tokens: int = DEFAULT_OUTPUT_TOKENS,
    tokenizer: str = DEFAULT_TOKENIZER,
    request_timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    output_dir: str | None = None,
    aiperf_args: list[str] | None = None,
) -> BenchmarkConfig:
    """Create a benchmark config from a Modal row and explicit overrides."""
    resolved_app_name = (app_name or (row.name if row else "") or "").strip()
    resolved_instance = (instance_name or (row.instance_name if row else "") or "").strip() or None
    resolved_url = (server_url or (row.web_url if row else "") or "").strip()
    if not resolved_url and username.strip() and resolved_app_name:
        from .backend import ModalBackend

        resolved_url = ModalBackend.default_server_url(username.strip(), app_name=resolved_app_name)

    resolved_model = (model_name or "").strip()
    if not resolved_model and row is not None:
        if backend == BackendType.VLLM:
            resolved_model = (
                (row.served_model_name or "").strip()
                or default_served_model_name(row.model_name)
            )
        else:
            resolved_model = (
                (row.served_model_name or "").strip()
                or default_llamacpp_served_model_name(row.repo_id, row.quant)
            )
    if not resolved_model:
        resolved_model = "default" if backend == BackendType.LLAMACPP else "llm"

    return BenchmarkConfig(
        backend=backend,
        app_name=resolved_app_name or None,
        instance_name=resolved_instance,
        server_url=normalize_aiperf_url(resolved_url),
        model_name=resolved_model,
        concurrency=concurrency or list(DEFAULT_CONCURRENCY),
        request_count=request_count,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tokenizer=(tokenizer or DEFAULT_TOKENIZER).strip() or DEFAULT_TOKENIZER,
        request_timeout_seconds=request_timeout_seconds,
        output_dir=output_dir,
        aiperf_args=list(aiperf_args or []),
    )


def merge_cached_benchmark_connections(rows: list[EndpointInfo]) -> None:
    """Merge persisted deployment connection summaries into Modal rows."""
    cache_path = SETTINGS_DIR / "deployment_connection_summaries.json"
    try:
        payload = json.loads(cache_path.read_text())
    except Exception:
        return
    entries = payload.get("entries")
    if not isinstance(entries, dict):
        return
    for row in rows:
        cached = entries.get((row.name or "").strip())
        if not isinstance(cached, dict):
            continue
        if not (row.web_url or "").strip():
            base_url = str(cached.get("base_url", "") or "").strip()
            if base_url:
                row.web_url = normalize_aiperf_url(base_url)
        if not (row.served_model_name or "").strip():
            model_id = str(cached.get("model_id", "") or "").strip()
            if model_id:
                row.served_model_name = model_id
        if not (row.display_name or "").strip():
            display_name = str(cached.get("display_name", "") or "").strip()
            if display_name:
                row.display_name = display_name


def build_aiperf_command(
    config: BenchmarkConfig,
    *,
    concurrency: int,
    artifact_dir: Path,
    executable: str = "aiperf",
) -> list[str]:
    """Build the AIPerf profile command for one concurrency value."""
    if not (config.server_url or "").strip():
        raise ValueError("Benchmark target server URL is required.")
    if not (config.model_name or "").strip():
        raise ValueError("Benchmark target model name is required.")

    cmd = [
        executable,
        "profile",
        "--model",
        str(config.model_name),
        "--url",
        normalize_aiperf_url(str(config.server_url)),
        "--endpoint-type",
        "chat",
        "--streaming",
        "--use-legacy-max-tokens",
        "--tokenizer",
        config.tokenizer,
        "--ui",
        "none",
        "--no-server-metrics",
        "--export-level",
        "summary",
        "--prompt-input-tokens-mean",
        str(config.input_tokens),
        "--output-tokens-mean",
        str(config.output_tokens),
        "--random-seed",
        str(DEFAULT_RANDOM_SEED),
        "--request-timeout-seconds",
        str(config.request_timeout_seconds),
        "--concurrency",
        str(concurrency),
        "--request-count",
        str(request_count_for_concurrency(concurrency, config.request_count)),
        "--artifact-dir",
        str(artifact_dir),
    ]
    cmd.extend(config.aiperf_args)
    return cmd


def expected_export_paths(artifact_dir: Path) -> tuple[Path, Path]:
    """Return default AIPerf JSON and CSV summary export paths for a run."""
    return artifact_dir / "profile_export_aiperf.json", artifact_dir / "profile_export_aiperf.csv"


def parse_aiperf_summary(json_path: Path, csv_path: Path) -> tuple[dict[str, float | None], str]:
    """Parse AIPerf exports, preferring JSON and falling back to CSV."""
    if json_path.exists():
        try:
            payload = json.loads(json_path.read_text())
            return _extract_metrics_from_json(payload), str(json_path)
        except Exception:
            pass
    if csv_path.exists():
        return _extract_metrics_from_csv(csv_path), str(csv_path)
    raise FileNotFoundError(f"No AIPerf summary export found in {json_path.parent}")


def aiperf_metrics_have_successful_requests(metrics: dict[str, float | None]) -> bool:
    """Return True when parsed AIPerf metrics include evidence of completed requests."""
    request_count = metrics.get("request_count")
    if request_count is not None and request_count <= 0:
        return False
    required_metric_keys = (
        "output_token_throughput",
        "request_throughput",
        "time_to_first_token_avg",
        "request_latency_avg",
    )
    return any(metrics.get(key) is not None for key in required_metric_keys)


def build_run_summary(config: BenchmarkConfig, run_dir: Path, results: list[BenchmarkConcurrencyResult]) -> BenchmarkRunSummary:
    """Build aggregate summary for a completed benchmark sweep."""
    best_result: BenchmarkConcurrencyResult | None = None
    best_value: float | None = None
    for result in results:
        value = result.metrics.get("output_token_throughput")
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_value = value
            best_result = result
    return BenchmarkRunSummary(
        config=config,
        run_dir=str(run_dir),
        results=results,
        success=all(result.success for result in results),
        best_concurrency=best_result.concurrency if best_result else None,
        best_output_token_throughput=best_value,
    )


def write_run_summary(summary: BenchmarkRunSummary) -> Path:
    """Persist a benchmark run summary JSON file."""
    run_dir = Path(summary.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / SUMMARY_FILENAME
    summary_path.write_text(json.dumps(benchmark_summary_to_dict(summary), indent=2, sort_keys=True))
    return summary_path


def benchmark_summary_to_dict(summary: BenchmarkRunSummary) -> dict[str, Any]:
    """Convert a benchmark run summary to JSON-serializable data."""
    return {
        "config": asdict(summary.config),
        "run_dir": summary.run_dir,
        "success": summary.success,
        "best_concurrency": summary.best_concurrency,
        "best_output_token_throughput": summary.best_output_token_throughput,
        "results": [asdict(result) for result in summary.results],
    }


def format_benchmark_summary(summary: BenchmarkRunSummary) -> list[str]:
    """Render a compact text summary sorted by concurrency."""
    lines = ["Benchmark results:"]
    sorted_results = sorted(summary.results, key=lambda result: result.concurrency)
    for result in sorted_results:
        metrics = result.metrics
        throughput = _format_optional(metrics.get("output_token_throughput"))
        request_rate = _format_optional(metrics.get("request_throughput"))
        ttft_p90 = _format_optional(metrics.get("time_to_first_token_p90"))
        latency_p90 = _format_optional(metrics.get("request_latency_p90"))
        status = "ok" if result.success else "failed"
        lines.append(
            f"  c={result.concurrency:<3} {status:<6} "
            f"out_tok/s={throughput:<8} req/s={request_rate:<8} "
            f"ttft_p90_ms={ttft_p90:<8} latency_p90_ms={latency_p90:<8}"
        )
    if summary.best_concurrency is not None and summary.best_output_token_throughput is not None:
        lines.append(
            "Best measured output throughput: "
            f"c={summary.best_concurrency} "
            f"({summary.best_output_token_throughput:.2f} tokens/sec)."
        )
    lines.append(f"Summary: {Path(summary.run_dir) / SUMMARY_FILENAME}")
    return lines


def _format_optional(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _extract_metrics_from_json(payload: Any) -> dict[str, float | None]:
    metrics = _empty_metrics()
    for metric_name in ("output_token_throughput", "request_throughput", "request_count"):
        source = _find_metric_payload(payload, metric_name)
        metrics[metric_name] = _extract_stat(source, ("avg", "value", "total", "count"))
    for metric_name in ("time_to_first_token", "inter_token_latency", "request_latency"):
        source = _find_metric_payload(payload, metric_name)
        if metric_name == "inter_token_latency":
            stats = ("avg", "p90")
        elif metric_name == "request_latency":
            stats = ("avg", "p90", "p99")
        else:
            stats = ("avg", "p50", "p90", "p99")
        for stat in stats:
            metrics[f"{metric_name}_{stat}"] = _extract_stat(source, (stat,))
    return metrics


def _extract_metrics_from_csv(csv_path: Path) -> dict[str, float | None]:
    metrics = _empty_metrics()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metric_label = _first_present(row, ("Metric", "metric", "name", "Metric Name"))
            metric_name = _metric_name_from_label(metric_label)
            if metric_name is None:
                continue
            if metric_name in {"output_token_throughput", "request_throughput", "request_count"}:
                metrics[metric_name] = _coerce_float(
                    _first_present(row, ("avg", "value", "Value", "mean"))
                )
                continue
            for stat in ("avg", "p50", "p90", "p99"):
                key = f"{metric_name}_{stat}"
                if key in metrics:
                    metrics[key] = _coerce_float(_first_present(row, (stat, stat.upper())))
    return metrics


def _empty_metrics() -> dict[str, float | None]:
    return {key: None for key in METRIC_KEYS}


def _find_metric_payload(payload: Any, metric_name: str) -> Any:
    aliases = _METRIC_ALIASES[metric_name]
    found = _find_metric_payload_inner(payload, aliases)
    return found


def _find_metric_payload_inner(payload: Any, aliases: tuple[str, ...]) -> Any:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if _normalize_key(str(key)) in aliases:
                return value
        label = _first_present(payload, ("metric", "Metric", "name", "label"))
        if _metric_name_from_label(label) is not None:
            normalized = _normalize_key(label)
            if normalized in aliases:
                return payload
        for value in payload.values():
            found = _find_metric_payload_inner(value, aliases)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_metric_payload_inner(value, aliases)
            if found is not None:
                return found
    return None


def _extract_stat(payload: Any, stat_names: tuple[str, ...]) -> float | None:
    if payload is None:
        return None
    if isinstance(payload, (int, float, str)) and "value" in stat_names:
        return _coerce_float(payload)
    if isinstance(payload, dict):
        for stat in stat_names:
            for key, value in payload.items():
                if _normalize_key(str(key)) == _normalize_key(stat):
                    nested = _extract_value(value)
                    if nested is not None:
                        return nested
        if "value" in stat_names:
            return _extract_value(payload)
    return None


def _extract_value(payload: Any) -> float | None:
    if isinstance(payload, (int, float, str)):
        return _coerce_float(payload)
    if isinstance(payload, dict):
        for key in ("value", "Value", "avg", "mean"):
            if key in payload:
                parsed = _coerce_float(payload[key])
                if parsed is not None:
                    return parsed
    return None


def _metric_name_from_label(label: str) -> str | None:
    normalized = _normalize_key(label)
    for metric_name, aliases in _METRIC_ALIASES.items():
        if normalized in aliases:
            return metric_name
    for metric_name, aliases in _METRIC_ALIASES.items():
        if any(alias and alias in normalized for alias in aliases):
            return metric_name
    return None


def _normalize_key(value: str) -> str:
    text = (value or "").casefold()
    text = text.replace("/sec", "persec")
    text = re.sub(r"\([^)]*\)", "", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return str(value)
    return ""


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text or text.upper() == "N/A":
        return None
    try:
        return float(text)
    except ValueError:
        return None
