from __future__ import annotations

import unittest
from unittest.mock import patch

from llm_launchpad.core.hf_auth import (
    _extract_username,
    get_huggingface_auth_status,
)


class _LocalTokenNotFoundError(Exception):
    pass


class _InvalidTokenError(Exception):
    pass


class HuggingFaceAuthTests(unittest.TestCase):
    def test_extract_username_reads_common_whoami_shapes(self) -> None:
        self.assertEqual(_extract_username({"name": "alice"}), "alice")
        self.assertEqual(_extract_username({"user": {"name": "bob"}}), "bob")
        self.assertEqual(_extract_username({"auth": {"username": "carol"}}), "carol")

    @patch("llm_launchpad.core.hf_auth._has_local_token", return_value=False)
    def test_get_huggingface_auth_status_returns_unauthenticated_without_token(self, _mock_token) -> None:
        status = get_huggingface_auth_status()
        self.assertFalse(status.authenticated)
        self.assertIsNone(status.username)
        self.assertIsNone(status.error)

    @patch("llm_launchpad.core.hf_auth._has_local_token", return_value=True)
    @patch("huggingface_hub.HfApi")
    def test_get_huggingface_auth_status_returns_username_on_success(self, mock_hf_api, _mock_token) -> None:  # type: ignore[no-untyped-def]
        mock_hf_api.return_value.whoami.return_value = {"name": "alice"}

        status = get_huggingface_auth_status()

        self.assertTrue(status.authenticated)
        self.assertEqual(status.username, "alice")
        self.assertIsNone(status.error)

    @patch("llm_launchpad.core.hf_auth._has_local_token", return_value=True)
    @patch("huggingface_hub.HfApi")
    def test_get_huggingface_auth_status_handles_missing_login(self, mock_hf_api, _mock_token) -> None:  # type: ignore[no-untyped-def]
        mock_hf_api.return_value.whoami.side_effect = _LocalTokenNotFoundError("Run `hf auth login`")

        status = get_huggingface_auth_status()

        self.assertFalse(status.authenticated)
        self.assertIsNone(status.username)
        self.assertIsNone(status.error)

    @patch("llm_launchpad.core.hf_auth._has_local_token", return_value=True)
    @patch("huggingface_hub.HfApi")
    def test_get_huggingface_auth_status_handles_invalid_token(self, mock_hf_api, _mock_token) -> None:  # type: ignore[no-untyped-def]
        mock_hf_api.return_value.whoami.side_effect = _InvalidTokenError("401 Unauthorized: Invalid user token")

        status = get_huggingface_auth_status()

        self.assertFalse(status.authenticated)
        self.assertEqual(status.error, "Invalid Hugging Face token")


if __name__ == "__main__":
    unittest.main()
