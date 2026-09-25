"""The on-machine idle watchdog, run for real against a fake runtime."""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from llm_launchpad.core.idle_watchdog import (
    WatchdogSpec,
    describe_idle_shutdown,
    prime_destroy_command,
    resolve_idle_shutdown,
    vast_destroy_command,
    watchdog_script,
)
from llm_launchpad.protocol.models import DeploymentConfig, LaunchpadSettings

# conftest replaces subprocess.Popen to catch provider CLIs; sh is not one.
_POPEN = subprocess.Popen

pytestmark = pytest.mark.skipif(
    shutil.which("sh") is None or shutil.which("curl") is None,
    reason="needs sh and curl",
)


class _Runtime:
    """Serves /metrics with counters the test moves by hand."""

    def __init__(self) -> None:
        self.tokens = 0
        self.running = 0
        self.up = True
        self.authorized: list[str] = []
        runtime = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                runtime.authorized.append(self.headers.get("Authorization", ""))
                if not runtime.up:
                    self.send_response(503)
                    self.end_headers()
                    return
                body = (
                    "# HELP llamacpp:tokens_predicted_total x\n"
                    f"llamacpp:tokens_predicted_total {runtime.tokens}\n"
                    f"llamacpp:requests_processing {runtime.running}\n"
                    "llamacpp:predicted_tokens_seconds 12.5\n"
                ).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: object) -> None:
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}/metrics"


def _start(tmp_path: Path, runtime: _Runtime, *, idle: int, grace: int = 3600) -> tuple[subprocess.Popen, Path]:
    marker = tmp_path / "destroyed"
    script = tmp_path / "watchdog.sh"
    script.write_text(watchdog_script(WatchdogSpec(
        metrics_url=runtime.url,
        endpoint_api_key="secret",
        idle_seconds=idle,
        destroy_command=f"touch {marker}",
        runtime_dir=str(tmp_path),
        startup_grace_seconds=grace,
        poll_seconds=1,
    )))
    process = _POPEN(["sh", str(script)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return process, marker


def _wait(predicate, timeout: float) -> bool:  # type: ignore[no-untyped-def]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.2)
    return predicate()


def test_activity_keeps_it_alive_and_silence_destroys_it(tmp_path: Path) -> None:
    runtime = _Runtime()
    process, marker = _start(tmp_path, runtime, idle=3)
    try:
        for _ in range(5):
            runtime.tokens += 10
            time.sleep(1)
        assert not marker.exists(), "destroyed while tokens were moving"
        runtime.running = 1  # a long prefill moves no counter
        time.sleep(4)
        assert not marker.exists(), "destroyed during an in-flight request"
        runtime.running = 0
        assert _wait(marker.exists, 8), "idle rental was never destroyed"
        assert process.wait(timeout=5) == 0
        assert runtime.authorized[0] == "Bearer secret"
        healthy, last_active, idle = (tmp_path / "idle-watchdog.state").read_text().split()
        assert (healthy, idle) == ("1", "3")
    finally:
        process.kill()
        runtime.server.shutdown()


def test_a_runtime_that_never_serves_is_destroyed_after_the_grace(tmp_path: Path) -> None:
    runtime = _Runtime()
    runtime.up = False
    process, marker = _start(tmp_path, runtime, idle=3600, grace=2)
    try:
        assert _wait(marker.exists, 8)
        output = process.communicate(timeout=5)[0]
        assert "never started serving" in output
    finally:
        process.kill()
        runtime.server.shutdown()


def test_the_idle_clock_starts_when_serving_starts(tmp_path: Path) -> None:
    # A 20-minute download must not count as 20 idle minutes.
    runtime = _Runtime()
    runtime.up = False
    process, marker = _start(tmp_path, runtime, idle=3)
    try:
        time.sleep(4)
        runtime.up = True
        time.sleep(2)
        assert not marker.exists()
        assert _wait(marker.exists, 8)
    finally:
        process.kill()
        runtime.server.shutdown()


@pytest.mark.parametrize(
    "command",
    [vast_destroy_command(), prime_destroy_command("https://api.example/api/v1", "pod-1", "/r/key", "/r/tunnel-id")],
)
def test_destroy_commands_are_valid_shell(command: str) -> None:
    script = watchdog_script(WatchdogSpec("http://x/metrics", "k", 60, command, "/r"))
    checker = _POPEN(["sh", "-n"], stdin=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _, errors = checker.communicate(script)
    assert checker.returncode == 0, errors


def test_vast_destroy_reads_the_instance_scoped_key_from_pid_1(tmp_path: Path) -> None:
    # Swap the network call for an echo to see which id and key it would use.
    command = vast_destroy_command().replace("curl -fsS -m 30 -X DELETE", "echo").replace(" >/dev/null", "")
    out = subprocess.run(
        ["sh", "-c", command],
        env={"PATH": "/usr/bin:/bin", "CONTAINER_ID": "42", "CONTAINER_API_KEY": "scoped"},
        capture_output=True, text=True, check=True,
    ).stdout
    assert "Bearer scoped" in out and "/instances/42/" in out


def test_settings_default_and_explicit_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    assert LaunchpadSettings().rental_idle_shutdown == 3600
    assert LaunchpadSettings.from_dict({"rental_idle_shutdown": "junk"}).rental_idle_shutdown == 3600
    assert LaunchpadSettings.from_dict({"rental_idle_shutdown": 0}).rental_idle_shutdown == 0
    assert resolve_idle_shutdown(DeploymentConfig(idle_shutdown_seconds=0)) == 0
    assert resolve_idle_shutdown(DeploymentConfig()) == 3600
    assert describe_idle_shutdown(3600) == "deleted after 1h with no requests"
    assert "bills until" in describe_idle_shutdown(0)
