"""Idle shutdown wiring for Prime pods and Vast rentals."""

from __future__ import annotations

import io
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from llm_launchpad.core import local_watchdog
from llm_launchpad.core.local_watchdog import IdleClock, activity_signature
from llm_launchpad.core.local_watchdog import run as run_local_watchdog
# Bound before conftest's autouse stub replaces the module attribute.
from llm_launchpad.core.local_watchdog import spawn_local_watchdog as real_spawn
from llm_launchpad.core.orchestrator import Orchestrator
from llm_launchpad.core.vast_deployment import VastDeploymentBackend
from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.models import DeploymentConfig, LaunchpadSettings

_METRICS = "llamacpp:tokens_predicted_total {tokens}\nllamacpp:requests_processing {running}\n"


def test_activity_signature_reads_counters_and_in_flight_gauges() -> None:
    counters, in_flight = activity_signature(
        'vllm:generation_tokens_total{model_name="m"} 12.0\n'
        'vllm:num_requests_running{model_name="m"} 1.0\n'
        "vllm:kv_cache_usage_perc 0.5\n"
    )
    assert counters == ('vllm:generation_tokens_total{model_name="m"} 12.0',)
    assert in_flight


def test_idle_clock_waits_for_serving_then_counts_silence() -> None:
    clock = IdleClock(idle_seconds=100, started_at=0)
    clock.observe(50, None)  # still downloading
    assert not clock.expired(500)
    clock.observe(500, _METRICS.format(tokens=1, running=0))
    clock.observe(560, _METRICS.format(tokens=5, running=0))
    assert not clock.expired(650)
    clock.observe(650, _METRICS.format(tokens=5, running=1))
    assert not clock.expired(700)
    clock.observe(700, _METRICS.format(tokens=5, running=0))
    assert clock.expired(800)


def test_local_watchdog_stops_an_idle_pod_and_exits_when_it_is_gone() -> None:
    now = [0.0]
    stopped: list[bool] = []
    response = SimpleNamespace(ok=True, text=_METRICS.format(tokens=3, running=0))
    with patch("requests.get", return_value=response):
        reason = run_local_watchdog(
            metrics_url="https://pod.example/metrics",
            endpoint_api_key="k",
            idle_seconds=120,
            pod_exists=lambda: True,
            stop=lambda: stopped.append(True),
            poll_seconds=60,
            clock=lambda: now[0],
            sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
        )
    assert reason == "stopped after idle window"
    assert stopped == [True]
    assert run_local_watchdog(
        metrics_url="x", endpoint_api_key="k", idle_seconds=1,
        pod_exists=lambda: False, stop=lambda: pytest.fail("stopped a gone pod"),
    ) == "pod is gone"


def test_spawn_hands_the_key_over_stdin_and_does_not_wait(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(local_watchdog, "LOCAL_WATCHDOG_LOG_DIR", tmp_path)
    process = SimpleNamespace(stdin=io.BytesIO())
    process.stdin.close = lambda: None  # keep the buffer readable
    popen = MagicMock(return_value=process)
    monkeypatch.setattr(local_watchdog.subprocess, "Popen", popen)
    real_spawn(
        app_name="qwen", backend="vllm", pod_id="pod-1",
        endpoint_url="https://t.example/v1", endpoint_api_key="secret", idle_seconds=3600,
    )
    argv = popen.call_args.args[0]
    assert "secret" not in argv
    assert argv[argv.index("--metrics-url") + 1] == "https://t.example/metrics"
    assert popen.call_args.kwargs["start_new_session"] is True
    assert process.stdin.getvalue() == b"secret\n"


def _orchestrator(settings: LaunchpadSettings) -> Orchestrator:
    store = MagicMock()
    store.load.return_value = settings
    return Orchestrator(config_store=store, backend=MagicMock(), prime_backend=MagicMock())


def _prime_config(**overrides: object) -> DeploymentConfig:
    return DeploymentConfig(
        backend=BackendType.VLLM, provider=ComputeProvider.PRIME, app_name="qwen",
        endpoint_api_key="endpoint-key", **overrides,  # type: ignore[arg-type]
    )


def _lines(events) -> list[str]:  # type: ignore[no-untyped-def]
    return [getattr(event, "line", "") for event in events]


def test_prime_without_opt_in_watches_from_this_computer(_stub_local_idle_watchdog) -> None:
    orch = _orchestrator(LaunchpadSettings())
    lines = _lines(orch._start_prime_idle_shutdown(_prime_config(), {}, "pod-1", "https://t.example"))
    orch.prime_backend.start_idle_watchdog.assert_not_called()
    assert _stub_local_idle_watchdog[0]["pod_id"] == "pod-1"
    assert _stub_local_idle_watchdog[0]["idle_seconds"] == 3600
    assert any("while this computer is awake" in line for line in lines)


def test_prime_with_opt_in_runs_on_the_pod(_stub_local_idle_watchdog) -> None:
    orch = _orchestrator(LaunchpadSettings(prime_self_terminate=True, rental_idle_shutdown=1800))
    lines = _lines(orch._start_prime_idle_shutdown(_prime_config(), {}, "pod-1", "https://t.example"))
    orch.prime_backend.start_idle_watchdog.assert_called_once_with({}, "pod-1", "endpoint-key", 1800)
    assert _stub_local_idle_watchdog == []
    assert any("runs on the pod" in line for line in lines)


def test_prime_on_pod_failure_falls_back_to_this_computer(_stub_local_idle_watchdog) -> None:
    orch = _orchestrator(LaunchpadSettings(prime_self_terminate=True))
    orch.prime_backend.start_idle_watchdog.side_effect = RuntimeError("no ssh")
    _lines(orch._start_prime_idle_shutdown(_prime_config(), {}, "pod-1", "https://t.example"))
    assert len(_stub_local_idle_watchdog) == 1


def test_idle_shutdown_off_starts_nothing(_stub_local_idle_watchdog) -> None:
    orch = _orchestrator(LaunchpadSettings())
    lines = _lines(orch._start_prime_idle_shutdown(
        _prime_config(idle_shutdown_seconds=0), {}, "pod-1", "https://t.example"
    ))
    assert _stub_local_idle_watchdog == []
    assert any("bills until you stop it" in line for line in lines)


def test_vast_uploads_and_detaches_the_watchdog() -> None:
    ssh = MagicMock()
    lines = _lines(VastDeploymentBackend._start_idle_watchdog(
        ssh, object(), DeploymentConfig(idle_shutdown_seconds=900), "endpoint-key"
    ))
    upload, start = ssh.run.call_args_list
    assert "idle-watchdog.sh" in upload.args[1]
    assert "IDLE=900" in upload.kwargs["input_text"]
    assert "CONTAINER_API_KEY" in upload.kwargs["input_text"]
    assert start.args[1].startswith("nohup sh ")
    assert any("deleted after 15m" in line for line in lines)


def test_vast_watchdog_failure_is_a_warning_not_a_failed_deploy() -> None:
    ssh = MagicMock()
    ssh.run.side_effect = RuntimeError("ssh dropped")
    lines = _lines(VastDeploymentBackend._start_idle_watchdog(
        ssh, object(), DeploymentConfig(idle_shutdown_seconds=900), "k"
    ))
    assert any("bills until you stop it" in line for line in lines)


@pytest.mark.parametrize(
    ("provider", "settings", "expected"),
    [
        (ComputeProvider.VAST, LaunchpadSettings(), "deleted after 1h with no requests"),
        (ComputeProvider.PRIME, LaunchpadSettings(), "while this computer is awake"),
        (ComputeProvider.PRIME, LaunchpadSettings(prime_self_terminate=True), "no requests["),
        (ComputeProvider.VAST, LaunchpadSettings(rental_idle_shutdown=0), "bills until you stop it"),
    ],
)
def test_confirm_screen_states_the_idle_stop(provider, settings, expected) -> None:  # type: ignore[no-untyped-def]
    from llm_launchpad.core.config import ConfigStore
    from llm_launchpad.protocol.enums import BillingModel
    from llm_launchpad.tui.screens.quick_deploy import _idle_shutdown_fact

    def plan(billing: BillingModel):  # type: ignore[no-untyped-def]
        return SimpleNamespace(quote=SimpleNamespace(provider=provider, billing_model=billing))

    ConfigStore().save(settings)
    assert expected in _idle_shutdown_fact(plan(BillingModel.PROVISIONED))
    assert _idle_shutdown_fact(plan(BillingModel.SCALE_TO_ZERO)) == ""


def test_cli_idle_shutdown_parses_durations_and_off() -> None:
    import typer

    from llm_launchpad.cli.main import _parse_idle_shutdown

    assert _parse_idle_shutdown(None) is None
    assert _parse_idle_shutdown("off") == 0
    assert _parse_idle_shutdown("45m") == 2700
    with pytest.raises(typer.BadParameter):
        _parse_idle_shutdown("later")
