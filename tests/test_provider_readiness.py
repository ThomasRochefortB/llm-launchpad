from __future__ import annotations

import unittest
from unittest.mock import patch

from llm_launchpad.core.modal_auth import ModalAuthStatus
from llm_launchpad.core.prime_auth import PrimeAuthStatus
from llm_launchpad.core.provider_readiness import (
    ProviderReadinessStage,
    check_modal_readiness,
    check_prime_readiness,
    check_vast_readiness,
    clear_provider_readiness_cache,
)
from llm_launchpad.core.vast_auth import VastCredentials
from llm_launchpad.protocol.models import VastAuthStatus
from llm_launchpad.tui.app import TuiApp


class ProviderReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        clear_provider_readiness_cache()

    def tearDown(self) -> None:
        clear_provider_readiness_cache()

    def test_modal_installed_but_unauthenticated_is_not_ready(self) -> None:
        with (
            patch("llm_launchpad.core.provider_readiness.ModalBackend.is_cli_available", return_value=True),
            patch(
                "llm_launchpad.core.provider_readiness.get_modal_auth_status",
                return_value=ModalAuthStatus(authenticated=False, error="not authenticated"),
            ),
        ):
            readiness = check_modal_readiness(verify=True, refresh=True)
        self.assertEqual(readiness.stage, ProviderReadinessStage.AUTH_FAILED)
        self.assertFalse(readiness.verified)
        self.assertFalse(readiness.has_credentials is False)  # failed auth still has local state to retry

    def test_modal_missing_cli_is_not_installed(self) -> None:
        with patch(
            "llm_launchpad.core.provider_readiness.ModalBackend.is_cli_available", return_value=False
        ):
            readiness = check_modal_readiness(verify=True, refresh=True)
        self.assertEqual(readiness.stage, ProviderReadinessStage.NOT_INSTALLED)
        self.assertFalse(readiness.has_credentials)

    def test_prime_key_present_is_unverified_until_live_check(self) -> None:
        from llm_launchpad.core.prime_auth import PrimeConfig

        with patch(
            "llm_launchpad.core.provider_readiness.load_prime_config",
            return_value=PrimeConfig(api_key="key"),
        ):
            local = check_prime_readiness(verify=False, refresh=True)
        self.assertEqual(local.stage, ProviderReadinessStage.CREDENTIALS_PRESENT)
        self.assertTrue(local.has_credentials)
        self.assertFalse(local.verified)

    def test_prime_invalid_key_is_auth_failed_not_unreachable(self) -> None:
        from llm_launchpad.core.prime_auth import PrimeConfig

        class _Backend:
            def preflight(self) -> tuple[bool, str]:
                return False, "Prime API authentication failed: bad key"

        with patch(
            "llm_launchpad.core.provider_readiness.load_prime_config",
            return_value=PrimeConfig(api_key="bad"),
        ):
            readiness = check_prime_readiness(verify=True, refresh=True, backend=_Backend())
        self.assertEqual(readiness.stage, ProviderReadinessStage.AUTH_FAILED)

    def test_prime_network_failure_is_unreachable(self) -> None:
        from llm_launchpad.core.prime_auth import PrimeConfig

        class _Backend:
            def preflight(self) -> tuple[bool, str]:
                return False, "Prime API request failed: connection timed out"

        with patch(
            "llm_launchpad.core.provider_readiness.load_prime_config",
            return_value=PrimeConfig(api_key="key"),
        ):
            readiness = check_prime_readiness(verify=True, refresh=True, backend=_Backend())
        self.assertEqual(readiness.stage, ProviderReadinessStage.UNREACHABLE)

    def test_vast_key_present_is_unverified_until_live_check(self) -> None:
        with patch(
            "llm_launchpad.core.provider_readiness.resolve_vast_credentials",
            return_value=VastCredentials("key", "stored"),
        ):
            local = check_vast_readiness(verify=False, refresh=True)
        self.assertEqual(local.stage, ProviderReadinessStage.CREDENTIALS_PRESENT)
        self.assertTrue(local.has_credentials)

    def test_vast_rejected_key_is_auth_failed(self) -> None:
        class _Backend:
            def auth_status(self) -> VastAuthStatus:
                return VastAuthStatus(False, source="stored", error="Vast rejected the API key or its permissions.")

        with patch(
            "llm_launchpad.core.provider_readiness.resolve_vast_credentials",
            return_value=VastCredentials("bad", "stored"),
        ):
            readiness = check_vast_readiness(verify=True, refresh=True, backend=_Backend())
        self.assertEqual(readiness.stage, ProviderReadinessStage.AUTH_FAILED)

    def test_tui_gate_rejects_installed_but_unauthenticated_modal(self) -> None:
        app = TuiApp(mouse_enabled=True)
        with (
            patch("llm_launchpad.tui.app.ModalBackend.is_cli_available", return_value=True),
            patch(
                "llm_launchpad.core.modal_auth.get_modal_auth_status",
                return_value=ModalAuthStatus(authenticated=False, error="not authenticated"),
            ),
            patch("llm_launchpad.tui.app.get_prime_auth_status", return_value=PrimeAuthStatus(authenticated=False)),
            patch("llm_launchpad.tui.app.resolve_vast_credentials", return_value=VastCredentials()),
        ):
            self.assertFalse(app._provider_is_configured())
