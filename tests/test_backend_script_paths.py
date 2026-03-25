from __future__ import annotations

import unittest

from llm_launchpad.core.backend import ModalBackend
from llm_launchpad.core.paths import MODAL_LLAMACPP_SCRIPT, MODAL_VLLM_SCRIPT
from llm_launchpad.protocol.enums import BackendType
from llm_launchpad.protocol.models import DeploymentConfig


class BackendScriptPathTests(unittest.TestCase):
    def test_backend_type_scripts_use_backend_module_paths(self) -> None:
        self.assertEqual(BackendType.VLLM.script, MODAL_VLLM_SCRIPT)
        self.assertEqual(BackendType.LLAMACPP.script, MODAL_LLAMACPP_SCRIPT)

    def test_vllm_run_command_uses_backend_module_path(self) -> None:
        cmd = ModalBackend.build_run_command(DeploymentConfig(backend=BackendType.VLLM))
        self.assertEqual(cmd, ["modal", "run", "-m", MODAL_VLLM_SCRIPT])

    def test_llamacpp_run_command_uses_backend_module_path_with_main(self) -> None:
        cmd = ModalBackend.build_run_command(
            DeploymentConfig(backend=BackendType.LLAMACPP, preload=False)
        )
        self.assertEqual(cmd[0:4], ["modal", "run", "-m", f"{MODAL_LLAMACPP_SCRIPT}::main"])

    def test_deploy_command_uses_backend_module_path(self) -> None:
        cmd = ModalBackend.build_deploy_command(BackendType.LLAMACPP)
        self.assertEqual(cmd, ["modal", "deploy", "-m", MODAL_LLAMACPP_SCRIPT])


if __name__ == "__main__":
    unittest.main()
