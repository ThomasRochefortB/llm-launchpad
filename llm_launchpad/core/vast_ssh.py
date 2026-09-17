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


# Why a connection failed, told apart without ever echoing the transport's
# words. A rental that refuses the port is still installing its SSH server; a
# rental that denies the key has finished and will refuse forever. Those two
# need opposite decisions, and until now both arrived as one message.
SSH_REFUSED, SSH_REJECTED = "refused", "rejected"
SSH_HOST_KEY, SSH_UNREACHABLE = "host-key", "unreachable"
SSH_THROTTLED = "throttled"

_SSH_REASONS = (
    # Host key first, because a changed key also prints a denial line and the
    # two mean different things here. A Vast rental is reached through a
    # shared proxy that answers before the container's own sshd does, so the
    # key seen in the first seconds is not always the key the rental ends up
    # presenting. That is provisioning churn on a host we have not trusted
    # yet, not evidence about the deployment key.
    (SSH_HOST_KEY, re.compile(
        r"host key verification failed|remote host identification has changed"
        r"|key for .* has changed",
        re.I,
    )),
    # Before a denial: an authentication-attempt limit is a verdict on how
    # often we knocked, not on the key we offered. This loop polls a rental
    # for as long as it takes to come up, and the deploy path already notes
    # that hundreds of failed authentications look like an attack to Vast's
    # shared SSH proxy -- so reading the proxy's own rate limit as a rejected
    # key blames the rental for our polling and ends it while it still works.
    (SSH_THROTTLED, re.compile(
        r"too many authentication|max(imum)? authentication attempts",
        re.I,
    )),
    (SSH_REJECTED, re.compile(
        r"permission denied|no supported authentication",
        re.I,
    )),
    (SSH_REFUSED, re.compile(
        r"connection refused|connection closed|connection reset"
        r"|kex_exchange_identification|banner exchange|broken pipe",
        re.I,
    )),
)


def ssh_failure_reason(stderr: str) -> str:
    """Classify an OpenSSH failure by what it says about the host.

    Callers poll a rental that is already billing, so the distinction decides
    money: waiting out a refusal costs a few more minutes, while waiting out a
    denied key costs the whole deadline and then a second rental. Only an
    authentication denial is read as a decision; a refused port and a changed
    host key are both things a provisioning rental does on its way up.
    """
    for reason, pattern in _SSH_REASONS:
        if pattern.search(stderr or ""):
            return reason
    return SSH_UNREACHABLE


class VastSshError(RuntimeError):
    """An SSH failure that still says which kind it was, with no stderr in it."""

    def __init__(self, message: str, reason: str = SSH_UNREACHABLE) -> None:
        super().__init__(message)
        self.reason = reason


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
            raise VastSshError(
                "Vast SSH command could not complete. Check OpenSSH and connectivity."
            ) from None
        if result.returncode:
            # Never echo command arguments, stderr, or a credential-bearing
            # script: a runtime script carries the endpoint key, and arguments
            # carry paths. Diagnosing a host that refuses every connection
            # needs the transport's own words, so allow opting in explicitly.
            detail = ""
            if os.environ.get("LLM_LAUNCHPAD_SSH_DEBUG") == "1":
                detail = " " + " ".join((result.stderr or "").split())[:400]
            raise VastSshError(
                "Vast SSH command failed. Check connectivity and the saved host key." + detail,
                ssh_failure_reason(result.stderr or ""),
            )
        return result

    def run(
        self, instance: VastInstance, command: str, *, input_text: str | None = None,
        multiplex: bool = False,
    ) -> str:
        """Run one command. ``multiplex`` rides the tunnel instead of dialling.

        A polling caller runs this every few seconds for as long as a model
        takes to download, and each fresh dial is a new authentication against
        Vast's shared SSH proxy. Where the tunnel is already open the poll is a
        channel on it, so the proxy sees one connection for the whole deploy.
        """
        session = ["-S", str(self.control)] if multiplex and self.control.exists() else []
        return self._run(
            [*self.args(instance), *session, f"root@{instance.ssh_host}", command],
            input_text=input_text,
        ).stdout

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
