from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from llm_launchpad.core.vast_state import VastState
from llm_launchpad.protocol.models import VastDeploymentRecord


class VastStateSecretTests(unittest.TestCase):
    """A destroyed rental's key authenticates a host that no longer exists."""

    def setUp(self) -> None:
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.state = VastState(Path(temporary.name))
        self.record = VastDeploymentRecord(
            name="llp-vast-test", label="llp-vast-abc", account_id="12",
            offer_id="1001", machine_id="42", repo_id="acme/model-GGUF",
            quant="Q4_K_M", served_model_name="model", endpoint_api_key="private-key",
        )
        self.state.save(self.record)
        self.directory = self.state.directory(self.record.name)
        for filename in ("id_ed25519", "id_ed25519.pub", "known_hosts", "ssh"):
            (self.directory / filename).write_text("secret")

    def test_removing_a_record_takes_the_key_material_with_it(self) -> None:
        self.state.remove(self.record.name)
        self.assertIsNone(self.state.load(self.record.name))
        for filename in ("record.json", "id_ed25519", "id_ed25519.pub", "known_hosts", "ssh"):
            with self.subTest(filename=filename):
                self.assertFalse((self.directory / filename).exists())

    def test_the_lock_file_survives_so_mutual_exclusion_does(self) -> None:
        # remove() runs inside locked(); unlinking the file another process
        # waits on would let it recreate the path and hold a different inode.
        with self.state.locked(self.record.name):
            self.state.remove(self.record.name)
        self.assertTrue((self.directory / "lock").exists())

    def test_removing_twice_is_harmless(self) -> None:
        self.state.remove(self.record.name)
        self.state.remove(self.record.name)
        self.assertIsNone(self.state.load(self.record.name))

    def test_a_live_rental_keeps_its_key(self) -> None:
        # Only removal drops the key: connect() needs it while the rental runs.
        self.assertTrue((self.directory / "id_ed25519").exists())
        self.assertIsNotNone(self.state.load(self.record.name))


if __name__ == "__main__":
    unittest.main()
