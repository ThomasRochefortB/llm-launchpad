from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from llm_launchpad.core.quick_deploy_refresh import (
    _AAHttpError,
    _reset_artificial_analysis_auth_status_cache,
    _write_aa_cache,
    get_artificial_analysis_auth_status,
)


def _aa_payload(*, tier: str = "free") -> dict[str, object]:
    return {
        "tier": tier,
        "data": [
            {
                "id": "model",
                "name": "Example 8B",
                "slug": "example-8b",
                "evaluations": {"artificial_analysis_coding_index": 50},
            }
        ],
    }


class ArtificialAnalysisAuthTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_artificial_analysis_auth_status_cache()

    def tearDown(self) -> None:
        _reset_artificial_analysis_auth_status_cache()

    def test_missing_environment_key_is_unauthenticated_without_network(self) -> None:
        with patch.dict("os.environ", {}, clear=True), patch(
            "llm_launchpad.core.quick_deploy_refresh.fetch_artificial_analysis_models"
        ) as fetch:
            status = get_artificial_analysis_auth_status()

        self.assertFalse(status.authenticated)
        self.assertIsNone(status.error)
        fetch.assert_not_called()

    def test_valid_key_returns_tier_and_shares_cached_fetch(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "aa.json"
            with patch(
                "llm_launchpad.core.quick_deploy_refresh.fetch_artificial_analysis_models",
                return_value=_aa_payload(tier="pro"),
            ) as fetch:
                first = get_artificial_analysis_auth_status(
                    api_key="secret-key",
                    cache_path=cache_path,
                )
                second = get_artificial_analysis_auth_status(
                    api_key="secret-key",
                    cache_path=cache_path,
                )
            cache_text = cache_path.read_text(encoding="utf-8")

        self.assertTrue(first.authenticated)
        self.assertEqual(first.tier, "pro")
        self.assertEqual(second, first)
        fetch.assert_called_once_with("secret-key")
        self.assertNotIn("secret-key", cache_text)
        self.assertIn("key_fingerprint", cache_text)

    def test_matching_fresh_cache_authenticates_without_network(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "aa.json"
            _write_aa_cache(
                cache_path,
                _aa_payload(),
                fetched_at=datetime.now(timezone.utc),
                api_key="secret-key",
            )
            with patch(
                "llm_launchpad.core.quick_deploy_refresh.fetch_artificial_analysis_models"
            ) as fetch:
                status = get_artificial_analysis_auth_status(
                    api_key="secret-key",
                    cache_path=cache_path,
                )

        self.assertTrue(status.authenticated)
        self.assertEqual(status.tier, "free")
        fetch.assert_not_called()

    def test_invalid_key_returns_clear_error(self) -> None:
        with TemporaryDirectory() as temporary_directory, patch(
            "llm_launchpad.core.quick_deploy_refresh.fetch_artificial_analysis_models",
            side_effect=_AAHttpError(401),
        ):
            status = get_artificial_analysis_auth_status(
                api_key="invalid-key",
                cache_path=Path(temporary_directory) / "aa.json",
            )

        self.assertFalse(status.authenticated)
        self.assertEqual(status.error, "Invalid Artificial Analysis API key")

    def test_rotated_key_does_not_reuse_prior_key_authentication(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "aa.json"
            _write_aa_cache(
                cache_path,
                _aa_payload(),
                fetched_at=datetime.now(timezone.utc),
                api_key="old-key",
            )
            with patch(
                "llm_launchpad.core.quick_deploy_refresh.fetch_artificial_analysis_models",
                return_value=_aa_payload(tier="pro"),
            ) as fetch:
                status = get_artificial_analysis_auth_status(
                    api_key="new-key",
                    cache_path=cache_path,
                )

        self.assertTrue(status.authenticated)
        self.assertEqual(status.tier, "pro")
        fetch.assert_called_once_with("new-key")


if __name__ == "__main__":
    unittest.main()
