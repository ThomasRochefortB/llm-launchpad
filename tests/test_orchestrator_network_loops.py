from __future__ import annotations

import types
import unittest
from unittest.mock import patch

from llm_launchpad.core.orchestrator import Orchestrator
from llm_launchpad.protocol.enums import BackendType, ComputeProvider, DeploymentState, OperationType
from llm_launchpad.protocol.events import ErrorEvent, LogEvent, OperationCompleteEvent, StateChangeEvent


class _Response:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class _FakeStdout:
    """Simulates a text-mode subprocess stdout (iterable, line-based)."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def readline(self) -> str:
        if not self._lines:
            return ""
        return self._lines.pop(0)

    def __iter__(self):
        return self

    def __next__(self) -> str:
        line = self.readline()
        if not line:
            raise StopIteration
        return line


class _FakeProc:
    def __init__(self, lines: list[str]) -> None:
        self.stdout = _FakeStdout(lines)
        self.terminated = False
        self._running = True

    def terminate(self) -> None:
        self.terminated = True
        self._running = False

    def poll(self) -> int | None:
        return None if self._running else 0


class _FakeExitedProc(_FakeProc):
    def __init__(self, lines: list[str]) -> None:
        super().__init__(lines)
        self._running = False

    def terminate(self) -> None:
        self.terminated = True


class _SyncThread:
    """Runs the target function synchronously (in the calling thread)."""

    def __init__(self, *, target=None, daemon=None, **_kwargs):
        self._target = target

    def start(self) -> None:
        if self._target:
            self._target()


class OrchestratorNetworkLoopTests(unittest.TestCase):
    def test_warmup_llamacpp_requires_completion_payload_not_any_http_200(self) -> None:
        responses = iter(
            [
                _Response(200, "Server is starting..."),
                _Response(200, '{"id":"cmpl-1","object":"text_completion","choices":[{"text":"ok"}]}'),
            ]
        )
        post_calls = {"n": 0}

        def _post(*_args, **_kwargs):
            post_calls["n"] += 1
            return next(responses)

        fake_requests = types.SimpleNamespace(post=_post)
        clock = {"time": 0.0}

        def _fake_time() -> float:
            clock["time"] += 1.0
            return clock["time"]

        with patch.dict("sys.modules", {"requests": fake_requests}):
            with patch("llm_launchpad.core.warmup.time.time", side_effect=_fake_time):
                with patch(
                    "llm_launchpad.core.warmup.shutdown_event",
                    return_value=types.SimpleNamespace(wait=lambda **_kwargs: False),
                ):
                    with patch(
                        "llm_launchpad.core.warmup.ModalBackend.test_curl_command",
                        return_value="curl ok",
                    ):
                        events = list(
                            Orchestrator().warmup(
                                backend=BackendType.LLAMACPP,
                                server_url="https://example.modal.run",
                                timeout=10,
                                tail_logs=False,
                            )
                        )

        self.assertGreaterEqual(post_calls["n"], 2)
        self.assertTrue(
            any(
                isinstance(e, LogEvent) and e.line == "Server is ready!"
                for e in events
            )
        )
        self.assertTrue(
            any(
                isinstance(e, OperationCompleteEvent)
                and e.operation == OperationType.WARMUP
                and e.success
                for e in events
            )
        )

    def test_warmup_success_vllm(self) -> None:
        fake_requests = types.SimpleNamespace(get=lambda *_args, **_kwargs: _Response(200, "ok"))
        with patch.dict("sys.modules", {"requests": fake_requests}):
            with patch(
                "llm_launchpad.core.warmup.shutdown_event",
                return_value=types.SimpleNamespace(wait=lambda **_kwargs: False),
            ):
                with patch(
                    "llm_launchpad.core.warmup.ModalBackend.test_curl_command",
                    return_value="curl ok",
                ) as curl_mock:
                    events = list(
                        Orchestrator().warmup(
                            backend=BackendType.VLLM,
                            server_url="https://example.modal.run",
                            timeout=10,
                            tail_logs=False,
                            served_model_name="Qwen3-0.6B",
                        )
                    )
        curl_mock.assert_called_once_with(
            BackendType.VLLM,
            "https://example.modal.run",
            served_model_name="Qwen3-0.6B",
            api_key=None,
        )

        self.assertTrue(any(isinstance(e, StateChangeEvent) and e.current == DeploymentState.HEALTHY for e in events))
        self.assertTrue(
            any(
                isinstance(e, OperationCompleteEvent)
                and e.operation == OperationType.WARMUP
                and e.success
                for e in events
            )
        )

    def test_warmup_timeout_after_non_200_responses(self) -> None:
        fake_requests = types.SimpleNamespace(get=lambda *_args, **_kwargs: _Response(503, "not ready"))
        call_count = {"n": 0}

        def _fake_time():
            call_count["n"] += 1
            return 0.0 if call_count["n"] <= 5 else 999.0

        with patch.dict("sys.modules", {"requests": fake_requests}):
            with patch("llm_launchpad.core.warmup.time.time", side_effect=_fake_time):
                with patch(
                    "llm_launchpad.core.warmup.shutdown_event",
                    return_value=types.SimpleNamespace(wait=lambda **_kwargs: False),
                ):
                    events = list(
                        Orchestrator().warmup(
                            backend=BackendType.VLLM,
                            server_url="https://example.modal.run",
                            timeout=1,
                            tail_logs=False,
                        )
                    )

        self.assertTrue(any(isinstance(e, ErrorEvent) and "Timed out after 1s" in e.message for e in events))
        self.assertTrue(
            any(
                isinstance(e, OperationCompleteEvent)
                and e.operation == OperationType.WARMUP
                and not e.success
                for e in events
            )
        )

    def test_warmup_reports_modal_gpu_scheduling_queue_state(self) -> None:
        responses = iter(
            [
                _Response(
                    503,
                    (
                        "Function 'serve' (fu-abc) is waiting to be scheduled on a "
                        "GPU_A100_80GB worker. Relaxing requirements (gpus=4) may lead "
                        "to faster scheduling"
                    ),
                ),
                _Response(200, "ok"),
            ]
        )
        fake_requests = types.SimpleNamespace(get=lambda *_args, **_kwargs: next(responses))
        # Provide a time mock that advances steadily but stays under timeout.
        time_val = {"t": 0.0}

        def _fake_time():
            time_val["t"] += 0.1
            return time_val["t"]

        with patch.dict("sys.modules", {"requests": fake_requests}):
            with patch("llm_launchpad.core.warmup.time.time", side_effect=_fake_time):
                with patch(
                    "llm_launchpad.core.warmup.shutdown_event",
                    return_value=types.SimpleNamespace(wait=lambda **_kwargs: False),
                ):
                    with patch(
                        "llm_launchpad.core.warmup.ModalBackend.test_curl_command",
                        return_value="curl ok",
                    ):
                        events = list(
                            Orchestrator().warmup(
                                backend=BackendType.VLLM,
                                server_url="https://example.modal.run",
                                timeout=10,
                                tail_logs=False,
                            )
                        )

        self.assertTrue(
            any(
                isinstance(e, StateChangeEvent)
                and e.current == DeploymentState.QUEUED
                and "GPU_A100_80GB" in e.detail
                and "gpus=4" in e.detail
                for e in events
            )
        )
        self.assertTrue(
            any(
                isinstance(e, LogEvent)
                and "Waiting for GPU scheduling" in e.line
                and e.operation == OperationType.WARMUP
                for e in events
            )
        )
        self.assertTrue(any(isinstance(e, StateChangeEvent) and e.current == DeploymentState.HEALTHY for e in events))

    def test_warmup_with_log_tailing_drains_modal_logs(self) -> None:
        """Reader thread puts lines in the queue; warmup drains them."""
        fake_proc = _FakeProc(lines=["modal-log-line\n"])
        fake_requests = types.SimpleNamespace(get=lambda *_args, **_kwargs: _Response(200, "ok"))
        with patch.dict("sys.modules", {"requests": fake_requests}):
            with patch("llm_launchpad.core.warmup.ModalBackend.logs_follow_args", return_value=["--follow"]):
                with patch("llm_launchpad.core.warmup.subprocess.Popen", return_value=fake_proc) as popen_mock:
                    with patch("llm_launchpad.core.warmup.threading.Thread", _SyncThread):
                        with patch("llm_launchpad.core.warmup.os.environ", {}):
                            with patch(
                                "llm_launchpad.core.warmup.ModalBackend.test_curl_command",
                                return_value="curl ok",
                            ):
                                events = list(
                                    Orchestrator().warmup(
                                        backend=BackendType.VLLM,
                                        server_url="https://example.modal.run",
                                        timeout=10,
                                        tail_logs=True,
                                        app_name="vllm-test",
                                    )
                                )
        popen_mock.assert_called_once()
        self.assertEqual(
            popen_mock.call_args.args[0],
            ["modal", "app", "logs", "--follow", "vllm-test"],
        )

        self.assertTrue(
            any(
                isinstance(e, LogEvent)
                and e.line == "modal-log-line"
                and e.operation == OperationType.WARMUP
                for e in events
            )
        )

    def test_warmup_prime_skips_modal_log_tail_and_fallback(self) -> None:
        """Prime warmups must never launch Modal log tailing or its fallback."""
        fake_requests = types.SimpleNamespace(get=lambda *_args, **_kwargs: _Response(200, "ok"))

        def _fail_popen(*_args, **_kwargs):
            raise AssertionError("modal app logs must not be launched for Prime warmups")

        def _fail_run(*_args, **_kwargs):
            raise AssertionError("historical modal app logs must not run for Prime warmups")

        with patch.dict("sys.modules", {"requests": fake_requests}):
            with patch("llm_launchpad.core.warmup.subprocess.Popen", side_effect=_fail_popen):
                with patch("llm_launchpad.core.warmup.subprocess.run", side_effect=_fail_run):
                    with patch(
                        "llm_launchpad.core.warmup.shutdown_event",
                        return_value=types.SimpleNamespace(wait=lambda **_kwargs: False),
                    ):
                        with patch(
                            "llm_launchpad.core.warmup.ModalBackend.test_curl_command",
                            return_value="curl ok",
                        ):
                            events = list(
                                Orchestrator().warmup(
                                    backend=BackendType.VLLM,
                                    server_url="https://tunnel.example.com",
                                    timeout=10,
                                    tail_logs=True,
                                    provider=ComputeProvider.PRIME,
                                )
                            )

        self.assertTrue(
            any(
                isinstance(e, OperationCompleteEvent)
                and e.operation == OperationType.WARMUP
                and e.success
                for e in events
            )
        )

    def test_warmup_flushes_remaining_logs_after_logs_process_exit(self) -> None:
        """All lines from an exited process appear via the reader thread queue."""
        fake_proc = _FakeExitedProc(lines=["first\n", "second\n", "third\n"])
        fake_requests = types.SimpleNamespace(get=lambda *_args, **_kwargs: _Response(503, "not ready"))

        call_count = {"n": 0}

        def _fake_time():
            call_count["n"] += 1
            return 0.0 if call_count["n"] <= 8 else 999.0

        with patch.dict("sys.modules", {"requests": fake_requests}):
            with patch("llm_launchpad.core.warmup.ModalBackend.logs_follow_args", return_value=["--follow"]):
                with patch("llm_launchpad.core.warmup.subprocess.Popen", return_value=fake_proc):
                    with patch("llm_launchpad.core.warmup.threading.Thread", _SyncThread):
                        with patch("llm_launchpad.core.warmup.os.environ", {}):
                            with patch("llm_launchpad.core.warmup.time.time", side_effect=_fake_time):
                                with patch(
                                    "llm_launchpad.core.warmup.shutdown_event",
                                    return_value=types.SimpleNamespace(wait=lambda **_kwargs: False),
                                ):
                                    events = list(
                                        Orchestrator().warmup(
                                            backend=BackendType.VLLM,
                                            server_url="https://example.modal.run",
                                            timeout=1,
                                            tail_logs=True,
                                            app_name="vllm-test",
                                        )
                                    )

        warmup_logs = [
            e.line
            for e in events
            if isinstance(e, LogEvent) and e.operation == OperationType.WARMUP
        ]
        self.assertIn("first", warmup_logs)
        self.assertIn("second", warmup_logs)
        self.assertIn("third", warmup_logs)

    def test_fetch_historical_logs_yields_unseen_lines(self) -> None:
        """_fetch_historical_logs returns only lines not already in *seen*."""
        fake_run_result = types.SimpleNamespace(
            stdout="line-A\nline-B\nline-C\n", returncode=0
        )
        seen: set[str] = {"line-A"}
        with patch("llm_launchpad.core.warmup.subprocess.run", return_value=fake_run_result):
            with patch("llm_launchpad.core.warmup.os.environ", {}):
                events = list(Orchestrator._fetch_historical_logs("myapp", seen))

        lines = [e.line for e in events if isinstance(e, LogEvent)]
        # The header line plus the two new content lines
        self.assertEqual(len(lines), 3)
        self.assertIn("── Fetched historical container logs ──", lines)
        self.assertIn("line-B", lines)
        self.assertIn("line-C", lines)
        # line-A was already seen → not duplicated
        self.assertNotIn(
            "line-A",
            [line for line in lines if line != "── Fetched historical container logs ──"],
        )
        # Both new lines recorded in seen
        self.assertIn("line-B", seen)
        self.assertIn("line-C", seen)

    def test_fetch_historical_logs_handles_timeout(self) -> None:
        """_fetch_historical_logs captures partial output on timeout."""
        import subprocess as _sp

        exc = _sp.TimeoutExpired(cmd=["modal"], timeout=15)
        exc.stdout = "partial-line\n"
        with patch("llm_launchpad.core.warmup.subprocess.run", side_effect=exc):
            with patch("llm_launchpad.core.warmup.os.environ", {}):
                events = list(Orchestrator._fetch_historical_logs("myapp", set()))

        lines = [e.line for e in events if isinstance(e, LogEvent)]
        self.assertIn("partial-line", lines)

    def test_warmup_historical_fetch_after_stream_exit(self) -> None:
        """After the live stream exits, warmup re-runs modal app logs for persisted crash output."""
        # The live stream delivers one line, then exits.
        stream_proc = _FakeExitedProc(lines=["live-line\n"])
        fake_requests = types.SimpleNamespace(get=lambda *_args, **_kwargs: _Response(503, "not ready"))

        # The historical fetch (subprocess.run) returns the live line PLUS
        # a crash traceback that the stream missed.
        hist_result = types.SimpleNamespace(
            stdout="live-line\nTraceback (most recent call last):\n  crash here\n",
            returncode=0,
        )

        call_count = {"n": 0}

        def _fake_time():
            call_count["n"] += 1
            return 0.0 if call_count["n"] <= 10 else 999.0

        with patch.dict("sys.modules", {"requests": fake_requests}):
            with patch("llm_launchpad.core.warmup.ModalBackend.logs_follow_args", return_value=[]):
                with patch("llm_launchpad.core.warmup.subprocess.Popen", return_value=stream_proc):
                    with patch("llm_launchpad.core.warmup.threading.Thread", _SyncThread):
                        with patch("llm_launchpad.core.warmup.subprocess.run", return_value=hist_result):
                            with patch("llm_launchpad.core.warmup.os.environ", {}):
                                with patch("llm_launchpad.core.warmup.time.time", side_effect=_fake_time):
                                    with patch(
                                        "llm_launchpad.core.warmup.shutdown_event",
                                        return_value=types.SimpleNamespace(wait=lambda **_kwargs: False),
                                    ):
                                        events = list(
                                            Orchestrator().warmup(
                                                backend=BackendType.VLLM,
                                                server_url="https://example.modal.run",
                                                timeout=1,
                                                tail_logs=True,
                                                app_name="vllm-test",
                                            )
                                        )

        warmup_logs = [
            e.line
            for e in events
            if isinstance(e, LogEvent) and e.operation == OperationType.WARMUP
        ]
        # Live stream line appears
        self.assertIn("live-line", warmup_logs)
        # Historical fetch header appears
        self.assertIn("── Fetched historical container logs ──", warmup_logs)
        # Crash traceback (only from historical fetch) appears
        self.assertIn("Traceback (most recent call last):", warmup_logs)
        self.assertIn("  crash here", warmup_logs)

    def test_check_status_success(self) -> None:
        fake_requests = types.SimpleNamespace(get=lambda *_args, **_kwargs: _Response(200, "ok"))
        with patch.dict("sys.modules", {"requests": fake_requests}):
            with patch("llm_launchpad.core.orchestrator.ModalBackend.test_curl_command", return_value="curl ok"):
                events = list(
                    Orchestrator().check_status(
                        backend=BackendType.VLLM,
                        server_url="https://example.modal.run",
                        timeout=5,
                    )
                )
        self.assertTrue(
            any(
                isinstance(e, OperationCompleteEvent)
                and e.operation == OperationType.STATUS
                and e.success
                for e in events
            )
        )

    def test_check_status_llamacpp_uses_served_model_name_in_probe_and_test_command(self) -> None:
        captured_payloads: list[str] = []
        captured_urls: list[str] = []

        def _post(url, **kwargs):
            captured_urls.append(str(url))
            captured_payloads.append(str(kwargs.get("data", "")))
            return _Response(200, '{"choices":[{"text":"ok"}]}')

        fake_requests = types.SimpleNamespace(post=_post)
        with patch.dict("sys.modules", {"requests": fake_requests}):
            with patch(
                "llm_launchpad.core.orchestrator.ModalBackend.test_curl_command",
                return_value="curl ok",
            ) as curl_mock:
                events = list(
                            Orchestrator().check_status(
                                backend=BackendType.LLAMACPP,
                                server_url="https://example.modal.run/v1",
                                timeout=5,
                                served_model_name="Nanbeige4.1-3B-Q4_K_M-GGUF",
                            )
                )

        self.assertEqual(len(captured_payloads), 1)
        self.assertEqual(captured_urls, ["https://example.modal.run/v1/completions"])
        self.assertIn('"model": "Nanbeige4.1-3B-Q4_K_M-GGUF"', captured_payloads[0])
        curl_mock.assert_called_once_with(
            BackendType.LLAMACPP,
            "https://example.modal.run/v1",
            served_model_name="Nanbeige4.1-3B-Q4_K_M-GGUF",
            api_key=None,
        )
        self.assertTrue(
            any(
                isinstance(e, OperationCompleteEvent)
                and e.operation == OperationType.STATUS
                and e.success
                for e in events
            )
        )

    def test_check_status_reports_modal_gpu_scheduling_queue_state(self) -> None:
        responses = iter(
            [
                _Response(
                    503,
                    (
                        "Function 'serve' (fu-abc) is waiting to be scheduled on a "
                        "GPU_A100_80GB worker. Relaxing requirements (gpus=4) may lead "
                        "to faster scheduling"
                    ),
                ),
                _Response(200, "ok"),
            ]
        )
        fake_requests = types.SimpleNamespace(get=lambda *_args, **_kwargs: next(responses))
        with patch.dict("sys.modules", {"requests": fake_requests}):
            with patch(
                "llm_launchpad.core.orchestrator.shutdown_event",
                return_value=types.SimpleNamespace(wait=lambda **_kwargs: False),
            ):
                with patch("llm_launchpad.core.orchestrator.ModalBackend.test_curl_command", return_value="curl ok"):
                    events = list(
                        Orchestrator().check_status(
                            backend=BackendType.VLLM,
                            server_url="https://example.modal.run",
                            timeout=5,
                        )
                    )
        self.assertTrue(
            any(
                isinstance(e, StateChangeEvent)
                and e.current == DeploymentState.QUEUED
                and "GPU_A100_80GB" in e.detail
                for e in events
            )
        )
        self.assertTrue(
            any(
                isinstance(e, LogEvent)
                and "Waiting for GPU scheduling" in e.line
                and e.operation == OperationType.STATUS
                for e in events
            )
        )
        self.assertTrue(
            any(
                isinstance(e, OperationCompleteEvent)
                and e.operation == OperationType.STATUS
                and e.success
                for e in events
            )
        )

    def test_check_status_timeout(self) -> None:
        fake_requests = types.SimpleNamespace(get=lambda *_args, **_kwargs: _Response(503, "nope"))
        with patch.dict("sys.modules", {"requests": fake_requests}):
            with patch("llm_launchpad.core.orchestrator.time.time", side_effect=[0.0, 0.1, 999.0]):
                with patch(
                    "llm_launchpad.core.orchestrator.shutdown_event",
                    return_value=types.SimpleNamespace(wait=lambda **_kwargs: False),
                ):
                    events = list(
                        Orchestrator().check_status(
                            backend=BackendType.VLLM,
                            server_url="https://example.modal.run",
                            timeout=1,
                        )
                    )

        self.assertTrue(any(isinstance(e, ErrorEvent) and "Unhealthy (timed out)" in e.message for e in events))
        self.assertTrue(
            any(
                isinstance(e, OperationCompleteEvent)
                and e.operation == OperationType.STATUS
                and not e.success
                for e in events
            )
        )

    def test_check_status_error_flow_with_request_exception(self) -> None:
        def _raise(*_args, **_kwargs):
            raise RuntimeError("network down")

        fake_requests = types.SimpleNamespace(get=_raise)
        with patch.dict("sys.modules", {"requests": fake_requests}):
            with patch("llm_launchpad.core.orchestrator.time.time", side_effect=[0.0, 0.1, 999.0]):
                with patch(
                    "llm_launchpad.core.orchestrator.shutdown_event",
                    return_value=types.SimpleNamespace(wait=lambda **_kwargs: False),
                ):
                    events = list(
                        Orchestrator().check_status(
                            backend=BackendType.VLLM,
                            server_url="https://example.modal.run",
                            timeout=1,
                        )
                    )

        self.assertTrue(any(isinstance(e, ErrorEvent) and "network down" in e.message for e in events))

    def test_warmup_missing_requests_emits_failed_completion(self) -> None:
        import builtins

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "requests":
                raise ImportError("requests not installed")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_fake_import):
            events = list(
                Orchestrator().warmup(
                    backend=BackendType.VLLM,
                    server_url="https://example.modal.run",
                    timeout=1,
                    tail_logs=False,
                )
            )

        self.assertTrue(any(isinstance(e, ErrorEvent) and "requests" in e.message for e in events))
        self.assertTrue(
            any(
                isinstance(e, OperationCompleteEvent)
                and e.operation == OperationType.WARMUP
                and not e.success
                and e.exit_code == 1
                for e in events
            )
        )

    def test_check_status_missing_requests_emits_failed_completion(self) -> None:
        import builtins

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "requests":
                raise ImportError("requests not installed")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_fake_import):
            events = list(
                Orchestrator().check_status(
                    backend=BackendType.VLLM,
                    server_url="https://example.modal.run",
                    timeout=1,
                )
            )

        self.assertTrue(any(isinstance(e, ErrorEvent) and "requests" in e.message for e in events))
        self.assertTrue(
            any(
                isinstance(e, OperationCompleteEvent)
                and e.operation == OperationType.STATUS
                and not e.success
                and e.exit_code == 1
                for e in events
            )
        )


if __name__ == "__main__":
    unittest.main()
