"""Dedicated keys and persistent loopback-only SSH forwards for Vast."""

import os
from pathlib import Path
import re
import shlex
import socket
import subprocess

from ..protocol.models import VastInstance


def ssh_key_startup(public_key: str, *, root: str = "/root") -> str:
    """Install the rental's public key after Vast initializes the container."""
    if not re.fullmatch(r"ssh-ed25519 [A-Za-z0-9+/=]+(?: [^\r\n]+)?", public_key):
        raise ValueError("A single Ed25519 public key is required for Vast startup.")
    directory = shlex.quote(root + "/.ssh")
    authorized = shlex.quote(root + "/.ssh/authorized_keys")
    key = shlex.quote(public_key)
    # Only the public key goes in instance metadata. Appending preserves keys
    # installed by Vast, and a leading newline handles files without one.
    return (
        f"set -eu; umask 077; mkdir -p {directory}; "
        f"chmod go-w {shlex.quote(root)}; chmod 700 {directory}; "
        f"touch {authorized}; chmod 600 {authorized}; "
        f"if ! grep -qxF -- {key} {authorized}; then "
        f"printf '\\n%s\\n' {key} >> {authorized}; fi"
    )


class VastSsh:
    """Use OpenSSH control sockets so endpoints survive the launching process."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.key = directory / "id_ed25519"
        self.control = directory / "ssh"

    def public_key(self) -> str:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not self.key.exists():
            self._run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(self.key)])
        result = self._run(["ssh-keygen", "-y", "-f", str(self.key)])
        return result.stdout.strip()

    def args(self, instance: VastInstance) -> list[str]:
        if not re.fullmatch(r"ssh[0-9]+\.vast\.ai", instance.ssh_host) or not 1 <= instance.ssh_port <= 65535:
            raise ValueError("Vast has not supplied a valid proxy SSH address.")
        return [
            "ssh", "-F", "/dev/null", "-i", str(self.key), "-p", str(instance.ssh_port),
            "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={self.directory / 'known_hosts'}",
            "-o", f"HostKeyAlias=llp-vast-{instance.id}",
            "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3",
        ]

    @staticmethod
    def _run(args: list[str], *, input_text: str | None = None, timeout: int = 30) -> subprocess.CompletedProcess[str]:
        try:
            result = subprocess.run(args, input=input_text, capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            raise RuntimeError("Vast SSH command could not complete. Check OpenSSH and connectivity.") from None
        if result.returncode:
            # Never echo command arguments, stderr, or a credential-bearing
            # script: a runtime script carries the endpoint key, and arguments
            # carry paths. Diagnosing a host that refuses every connection
            # needs the transport's own words, so allow opting in explicitly.
            detail = ""
            if os.environ.get("LLM_LAUNCHPAD_SSH_DEBUG") == "1":
                detail = " " + " ".join((result.stderr or "").split())[:400]
            raise RuntimeError(
                "Vast SSH command failed. Check connectivity and the saved host key." + detail
            )
        return result

    def run(self, instance: VastInstance, command: str, *, input_text: str | None = None) -> str:
        return self._run([*self.args(instance), f"root@{instance.ssh_host}", command], input_text=input_text).stdout

    def connected(self, instance: VastInstance) -> bool:
        if not self.control.exists():
            return False
        try:
            self._run([*self.args(instance), "-S", str(self.control), "-O", "check", f"root@{instance.ssh_host}"])
        except (ValueError, RuntimeError):
            return False
        return True

    def connect(self, instance: VastInstance, port: int = 0) -> int:
        if self.connected(instance):
            if not port:
                raise ValueError("An existing Vast tunnel has no recorded local port.")
            return port
        self.control.unlink(missing_ok=True)
        if not port:
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
        if not 1 <= port <= 65535:
            raise ValueError("Invalid local tunnel port.")
        # A port race fails with ExitOnForwardFailure; never publish the URL
        # unless OpenSSH confirms it bound the requested loopback listener.
        self._run([
            *self.args(instance), "-M", "-S", str(self.control), "-f", "-N", "-T",
            "-o", "ControlPersist=yes", "-o", "ExitOnForwardFailure=yes",
            "-L", f"127.0.0.1:{port}:127.0.0.1:8000", f"root@{instance.ssh_host}",
        ])
        return port

    def disconnect(self, instance: VastInstance) -> None:
        if self.control.exists():
            self._run([*self.args(instance), "-S", str(self.control), "-O", "exit", f"root@{instance.ssh_host}"])
