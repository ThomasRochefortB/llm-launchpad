"""Regressions for the phantom "default" endpoint in the fleet.

Short-lived `modal run` helpers create a Modal app for the duration of the
call. Left unnamed they inherited the backend script's default, which is the
legacy app name, so fleet discovery claimed them as deployments and the
Manage Endpoints table listed an endpoint called "default" that nobody
deployed -- carrying a derived Base URL that never served anything.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from llm_launchpad.core.backend import _extract_modal_app_rows
from llm_launchpad.core.naming import (
    infer_backend_from_app_name,
    infer_instance_from_app_name,
    utility_app_name,
)
from llm_launchpad.core.opencode import visible_launchpad_rows
from llm_launchpad.core.orchestrator import Orchestrator
from llm_launchpad.protocol.enums import BackendType, ComputeProvider
from llm_launchpad.protocol.events import LogEvent, OperationCompleteEvent
from llm_launchpad.protocol.models import EndpointInfo
from llm_launchpad.tui.connection import resolve_openai_base_url


class UtilityAppNameTests(unittest.TestCase):
    def test_utility_names_are_not_claimed_as_deployments(self) -> None:
        for backend in (BackendType.LLAMACPP, BackendType.VLLM):
            with self.subTest(backend=backend):
                name = utility_app_name(backend)
                self.assertIsNone(infer_backend_from_app_name(name))
                self.assertIsNone(infer_instance_from_app_name(name, backend))

    def test_utility_names_are_distinct_per_backend(self) -> None:
        self.assertNotEqual(
            utility_app_name(BackendType.LLAMACPP),
            utility_app_name(BackendType.VLLM),
        )

    def test_helper_runs_do_not_become_fleet_endpoints(self) -> None:
        payload = [
            {"app_id": "ap-1", "description": utility_app_name(BackendType.LLAMACPP), "state": "stopped"},
            {"app_id": "ap-2", "description": utility_app_name(BackendType.LLAMACPP), "state": "stopped"},
            {"app_id": "ap-3", "description": "llamacpp-glm-5-3-flash", "state": "stopped"},
        ]
        rows = visible_launchpad_rows(_extract_modal_app_rows(payload))
        self.assertEqual([row.instance_name for row in rows], ["glm-5-3-flash"])

    def test_legacy_app_name_still_resolves_for_real_old_deployments(self) -> None:
        # Apps genuinely deployed under the pre-instance name must keep working.
        backend = infer_backend_from_app_name("llamacpp-server")
        self.assertEqual(backend, BackendType.LLAMACPP)
        self.assertEqual(infer_instance_from_app_name("llamacpp-server", backend), "default")


class UtilityRunEnvTests(unittest.TestCase):
    @patch("llm_launchpad.core.orchestrator.ModalBackend.run_modal_script_entrypoint_capture")
    def test_storage_listing_names_its_throwaway_app(self, mock_capture) -> None:  # type: ignore[no-untyped-def]
        mock_capture.return_value = None
        Orchestrator()._list_llamacpp_models_via_backend()

        env = mock_capture.call_args.kwargs["env"]
        self.assertEqual(env["MODAL_APP_NAME"], utility_app_name(BackendType.LLAMACPP))

    @patch("llm_launchpad.core.orchestrator.ModalBackend.run_modal_script_entrypoint")
    def test_predownload_names_its_throwaway_app(self, mock_run) -> None:  # type: ignore[no-untyped-def]
        for backend in (BackendType.LLAMACPP, BackendType.VLLM):
            with self.subTest(backend=backend):
                mock_run.return_value = iter(
                    [LogEvent(line="downloading"), OperationCompleteEvent(success=True, exit_code=0)]
                )
                list(Orchestrator().predownload_model(backend=backend, model_id="Qwen/Qwen3-4B"))

                env = mock_run.call_args.kwargs["env"]
                self.assertEqual(env["MODAL_APP_NAME"], utility_app_name(backend))


class DerivedBaseUrlTests(unittest.TestCase):
    @staticmethod
    def _row(state: str, web_url: str | None = None) -> EndpointInfo:
        return EndpointInfo(
            name="llamacpp-glm",
            app_id="ap-1",
            state=state,
            backend=BackendType.LLAMACPP,
            provider=ComputeProvider.MODAL,
            web_url=web_url,
        )

    def test_stopped_app_gets_no_invented_url(self) -> None:
        for state in ("stopped", "stopping", "terminated", "archived"):
            with self.subTest(state=state):
                url, derived = resolve_openai_base_url(self._row(state), username="someone")
                self.assertIsNone(url)
                self.assertFalse(derived)

    def test_starting_app_keeps_its_derived_url(self) -> None:
        url, derived = resolve_openai_base_url(self._row("deploying"), username="someone")
        self.assertEqual(url, "https://someone--llamacpp-glm-serve.modal.run/v1")
        self.assertTrue(derived)

    def test_reported_url_survives_a_terminal_state(self) -> None:
        # A provider that reports a real URL is believed regardless of state;
        # only the *invented* URL is withheld.
        url, derived = resolve_openai_base_url(
            self._row("stopped", web_url="https://someone--llamacpp-glm-serve.modal.run"),
            username="someone",
        )
        self.assertEqual(url, "https://someone--llamacpp-glm-serve.modal.run/v1")
        self.assertFalse(derived)


if __name__ == "__main__":
    unittest.main()
