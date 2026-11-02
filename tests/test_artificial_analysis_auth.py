from __future__ import annotations

from datetime import datetime, UTC
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

import hashlib
import json

from llm_launchpad.core import artificial_analysis_auth as auth

from llm_launchpad.core.artificial_analysis_auth import (
    clear_artificial_analysis_api_key,
    load_saved_artificial_analysis_api_key,
    resolve_artificial_analysis_api_key,
    save_artificial_analysis_api_key,
)
from llm_launchpad.core.artificial_analysis import (
    _AAHttpError,
    _load_aa_rankings,
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
        self.assertNotIn("key_fingerprint", cache_text)
        self.assertNotIn(hashlib.sha256(b"secret-key").hexdigest(), cache_text)

    def test_disk_cache_does_not_authenticate_a_key_after_restart(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "aa.json"
            _write_aa_cache(
                cache_path,
                _aa_payload(),
                fetched_at=datetime.now(UTC),
            )
            _reset_artificial_analysis_auth_status_cache()
            with patch(
                "llm_launchpad.core.artificial_analysis.fetch_artificial_analysis_models",
                return_value=_aa_payload(),
            ) as fetch:
                status = get_artificial_analysis_auth_status(
                    api_key="secret-key",
                    cache_path=cache_path,
                )

        self.assertTrue(status.authenticated)
        self.assertEqual(status.tier, "free")
        fetch.assert_called_once_with("secret-key")

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

    def test_verified_catalog_is_reused_within_the_same_process(self) -> None:
        with TemporaryDirectory() as directory, patch(
            "llm_launchpad.core.artificial_analysis.fetch_artificial_analysis_models",
            return_value=_aa_payload(),
        ) as fetch:
            path = Path(directory) / "cache.json"
            first = _load_aa_rankings(api_key="secret-key", cache_path=path)
            second = _load_aa_rankings(api_key="secret-key", cache_path=path)
        assert first is not None and second is not None
        self.assertEqual(first.freshness, "live")
        self.assertEqual(second.freshness, "cached")
        fetch.assert_called_once_with("secret-key")

    def test_cached_models_cannot_authenticate_an_invalid_key(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            _write_aa_cache(path, _aa_payload(), fetched_at=datetime.now(UTC))
            with patch(
                "llm_launchpad.core.artificial_analysis.fetch_artificial_analysis_models",
                side_effect=_AAHttpError(401),
            ):
                status = get_artificial_analysis_auth_status(api_key="invalid-key", cache_path=path)
        self.assertFalse(status.authenticated)

    def test_rotated_key_does_not_reuse_prior_key_authentication(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "aa.json"
            with patch(
                "llm_launchpad.core.artificial_analysis.fetch_artificial_analysis_models",
                side_effect=[_aa_payload(), _aa_payload(tier="pro")],
            ) as fetch:
                first = get_artificial_analysis_auth_status(api_key="old-key", cache_path=cache_path)
                second = get_artificial_analysis_auth_status(api_key="new-key", cache_path=cache_path)
        self.assertTrue(first.authenticated)
        self.assertTrue(second.authenticated)
        self.assertEqual(second.tier, "pro")
        self.assertEqual([call.args[0] for call in fetch.call_args_list], ["old-key", "new-key"])


class ArtificialAnalysisKeyPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.values: dict[tuple[str, str], str] = {}
        self.store = MagicMock()
        self.store.get_password.side_effect = lambda service, account: self.values.get((service, account))
        self.store.set_password.side_effect = lambda service, account, value: self.values.update({(service, account): value})
        self.store.delete_password.side_effect = lambda service, account: self.values.pop((service, account))
        patcher = patch.object(auth, "_credential_store", return_value=self.store)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_save_does_not_create_a_plaintext_file(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "aa.json"
            save_artificial_analysis_api_key("secret-key", path=path)
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])
            self.assertEqual(load_saved_artificial_analysis_api_key(path), "secret-key")

    def test_legacy_key_migrates_only_after_verified_secure_save(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "aa.json"
            path.write_text(json.dumps({"api_key": "legacy-key"}))
            temporary_path = path.with_suffix(".json.tmp")
            temporary_path.write_text(json.dumps({"api_key": "interrupted-key"}))
            self.assertEqual(load_saved_artificial_analysis_api_key(path), "legacy-key")
            self.assertFalse(path.exists())
            self.assertFalse(temporary_path.exists())
            self.store.set_password.assert_called_once_with(auth._SERVICE, auth._ACCOUNT, "legacy-key")

    def test_failed_migration_retains_legacy_file_without_using_its_key(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "aa.json"
            path.write_text(json.dumps({"api_key": "legacy-key"}))
            self.store.set_password.side_effect = RuntimeError("legacy-key")
            self.assertEqual(load_saved_artificial_analysis_api_key(path), "")
            self.assertTrue(path.exists())

    def test_failed_verification_retains_legacy_file(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "aa.json"
            path.write_text(json.dumps({"api_key": "legacy-key"}))
            self.store.set_password.side_effect = None
            self.assertEqual(load_saved_artificial_analysis_api_key(path), "")
            self.assertTrue(path.exists())

    def test_save_failure_has_no_plaintext_fallback_or_secret_in_error(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "aa.json"
            self.store.set_password.side_effect = RuntimeError("secret-key")
            with self.assertRaises(RuntimeError) as caught:
                save_artificial_analysis_api_key("secret-key", path=path)
            self.assertNotIn("secret-key", str(caught.exception))
            self.assertFalse(path.exists())

    def test_environment_key_does_not_access_credential_store(self) -> None:
        with patch.dict("os.environ", {auth.AAI_API_KEY_ENV: "env-key"}):
            self.assertEqual(resolve_artificial_analysis_api_key(), "env-key")
        self.store.get_password.assert_not_called()

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

class SecureCredentialBackendTests(unittest.TestCase):
    def test_plaintext_null_and_fail_backends_are_rejected(self) -> None:
        for module in ("keyrings.alt.file", "keyring.backends.null", "keyring.backends.fail"):
            backend = type("Backend", (), {"__module__": module})()
            with self.subTest(module=module), patch.object(auth.keyring, "get_keyring", return_value=backend):
                with self.assertRaises(RuntimeError):
                    auth._credential_store()

    def test_os_secret_service_is_accepted(self) -> None:
        backend = type("Keyring", (), {"__module__": "keyring.backends.SecretService"})()
        with patch.object(auth.keyring, "get_keyring", return_value=backend):
            self.assertIs(auth._credential_store(), backend)


    def test_chainer_skips_plaintext_backends(self) -> None:
        from keyring.backends.chainer import ChainerBackend

        plaintext = type("Keyring", (), {"__module__": "keyrings.alt.file"})()
        secure = type("Keyring", (), {"__module__": "keyring.backends.SecretService"})()
        with (
            patch.object(auth.keyring, "get_keyring", return_value=ChainerBackend()),
            patch.object(ChainerBackend, "backends", [plaintext, secure]),
        ):
            self.assertIs(auth._credential_store(), secure)


if __name__ == "__main__":
    unittest.main()
