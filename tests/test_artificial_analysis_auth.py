from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import stat

from llm_launchpad.core.artificial_analysis_auth import (
    clear_artificial_analysis_api_key,
    load_saved_artificial_analysis_api_key,
    resolve_artificial_analysis_api_key,
    save_artificial_analysis_api_key,
)
from llm_launchpad.core.artificial_analysis import (
    _AAHttpError,
    _reset_artificial_analysis_auth_status_cache,
    _write_aa_cache,
    get_artificial_analysis_auth_status,
)

AAI_AUTH_PATH_TARGET = "llm_launchpad.core.artificial_analysis_auth.AAI_AUTH_PATH"


def _aa_payload(*, tier: str = "free") -> dict[str, object]:
    return {
        "tier": tier,
        "data": [
            {
                "id": "model",
                "name": "Example 8B",
                "slug": "example-8b",
                "evaluations": {"artificial_analysis_intelligence_index": 50},
            }
        ],
    }


class ArtificialAnalysisAuthTests(unittest.TestCase):
    def setUp(self) -> None:
        _reset_artificial_analysis_auth_status_cache()

    def tearDown(self) -> None:
        _reset_artificial_analysis_auth_status_cache()

    def test_missing_environment_key_is_unauthenticated_without_network(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "llm_launchpad.core.artificial_analysis_auth.load_saved_artificial_analysis_api_key",
                return_value="",
            ),
            patch(
                "llm_launchpad.core.artificial_analysis.fetch_artificial_analysis_models"
            ) as fetch,
        ):
            status = get_artificial_analysis_auth_status()

        self.assertFalse(status.authenticated)
        self.assertIsNone(status.error)
        fetch.assert_not_called()

    def test_valid_key_returns_tier_and_shares_cached_fetch(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "aa.json"
            with patch(
                "llm_launchpad.core.artificial_analysis.fetch_artificial_analysis_models",
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
                fetched_at=datetime.now(UTC),
                api_key="secret-key",
            )
            with patch(
                "llm_launchpad.core.artificial_analysis.fetch_artificial_analysis_models"
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
            "llm_launchpad.core.artificial_analysis.fetch_artificial_analysis_models",
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
                fetched_at=datetime.now(UTC),
                api_key="old-key",
            )
            with patch(
                "llm_launchpad.core.artificial_analysis.fetch_artificial_analysis_models",
                return_value=_aa_payload(tier="pro"),
            ) as fetch:
                status = get_artificial_analysis_auth_status(
                    api_key="new-key",
                    cache_path=cache_path,
                )

        self.assertTrue(status.authenticated)
        self.assertEqual(status.tier, "pro")
        fetch.assert_called_once_with("new-key")


class ArtificialAnalysisKeyPersistenceTests(unittest.TestCase):
    def test_saved_key_file_is_owner_only(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            key_path = Path(temporary_directory) / "aa.json"
            save_artificial_analysis_api_key("secret-key", path=key_path)

            self.assertEqual(stat.S_IMODE(key_path.stat().st_mode), 0o600)

    def test_save_and_load_round_trip(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            key_path = Path(temporary_directory) / "aa.json"
            save_artificial_analysis_api_key("secret-key", path=key_path)

            self.assertEqual(load_saved_artificial_analysis_api_key(key_path), "secret-key")

    def test_load_missing_file_returns_empty(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            self.assertEqual(
                load_saved_artificial_analysis_api_key(Path(temporary_directory) / "aa.json"),
                "",
            )

    def test_clear_removes_stored_key(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            key_path = Path(temporary_directory) / "aa.json"
            save_artificial_analysis_api_key("secret-key", path=key_path)

            self.assertTrue(clear_artificial_analysis_api_key(key_path))
            self.assertFalse(key_path.exists())
            self.assertFalse(clear_artificial_analysis_api_key(key_path))

    def test_resolve_prefers_environment_over_stored_key(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            key_path = Path(temporary_directory) / "aa.json"
            save_artificial_analysis_api_key("stored-key", path=key_path)
            with (
                patch.dict("os.environ", {"ARTIFICIAL_ANALYSIS_API_KEY": "env-key"}),
                patch(AAI_AUTH_PATH_TARGET, key_path),
            ):
                self.assertEqual(resolve_artificial_analysis_api_key(), "env-key")

    def test_resolve_falls_back_to_stored_key(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            key_path = Path(temporary_directory) / "aa.json"
            save_artificial_analysis_api_key("stored-key", path=key_path)
            with (
                patch.dict("os.environ", {}, clear=True),
                patch(AAI_AUTH_PATH_TARGET, key_path),
            ):
                self.assertEqual(resolve_artificial_analysis_api_key(), "stored-key")

    def test_resolve_ignores_whitespace_only_environment_key(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            key_path = Path(temporary_directory) / "aa.json"
            save_artificial_analysis_api_key("stored-key", path=key_path)
            with (
                patch.dict(
                    "os.environ",
                    {"ARTIFICIAL_ANALYSIS_API_KEY": "  \t  "},
                    clear=True,
                ),
                patch(AAI_AUTH_PATH_TARGET, key_path),
            ):
                self.assertEqual(resolve_artificial_analysis_api_key(), "stored-key")

    def test_saved_key_is_used_by_default_auth_status(self) -> None:
        _reset_artificial_analysis_auth_status_cache()
        with TemporaryDirectory() as temporary_directory:
            key_path = Path(temporary_directory) / "aa.json"
            save_artificial_analysis_api_key("stored-key", path=key_path)
            with (
                patch.dict("os.environ", {}, clear=True),
                patch(AAI_AUTH_PATH_TARGET, key_path),
                patch(
                    "llm_launchpad.core.artificial_analysis.fetch_artificial_analysis_models",
                    return_value=_aa_payload(tier="pro"),
                ) as fetch,
            ):
                status = get_artificial_analysis_auth_status(
                    cache_path=Path(temporary_directory) / "cache.json"
                )

        _reset_artificial_analysis_auth_status_cache()
        self.assertTrue(status.authenticated)
        self.assertEqual(status.tier, "pro")
        fetch.assert_called_once_with("stored-key")


if __name__ == "__main__":
    unittest.main()
