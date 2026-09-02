from __future__ import annotations

from pathlib import Path
import stat
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from llm_launchpad.core.connection_store import (
    load_connection_entries,
    merge_connections,
    remove_connection,
    rows_from_connection_cache,
    save_connection,
)
from llm_launchpad.core.orchestrator import Orchestrator
from llm_launchpad.core.prime_auth import PrimeConfig, load_prime_config
from llm_launchpad.core.prime_backend import (
    PRIME_DEFAULT_BOOTSTRAP_IMAGE,
    PrimeApiError,
    PrimeBackend,
    PrimeDiskOffer,
    PrimeTunnel,
    default_prime_container_image,
    is_compatible_prime_offer,
    parse_prime_offer,
    preferred_prime_offer_image,
    prime_offer_gpu_memory_gb,
    resolve_prime_launch_spec,
    select_prime_offer,
)
from llm_launchpad.protocol.enums import BackendType, ComputeProvider, OperationType
from llm_launchpad.core.prime_disks import remember_prime_disk, StoredPrimeDisk
from llm_launchpad.core.provider_options import prime_provider_options
from llm_launchpad.protocol.events import EndpointAvailableEvent, LogEvent, OperationCompleteEvent
from llm_launchpad.protocol.models import (
    ComputeOffer,
    DeploymentConfig,
    EndpointInfo,
    PrimeProviderOptions,
    ReasoningCapabilities,
)


def _offer_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "cloudId": "n3-H100x1",
        "gpuType": "H100_80GB",
        "socket": "PCIe",
        "provider": "hyperstack",
        "region": "canada",
        "dataCenter": "CANADA-1",
        "country": "CA",
        "gpuCount": 1,
        "gpuMemory": 80,
        "disk": {"defaultCount": 100},
        "vcpu": {"defaultCount": 16},
        "memory": {"defaultCount": 180},
        "stockStatus": "Available",
        "security": "secure_cloud",
        "prices": {"onDemand": 1.9, "isVariable": False},
        "images": ["ubuntu_22_cuda_12"],
        "isSpot": False,
    }
    payload.update(overrides)
    return payload


class PrimeAuthTests(unittest.TestCase):
    def test_load_config_uses_environment_before_prime_cli_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text('{"api_key":"file-key","team_id":"file-team"}')
            with patch.dict(
                "os.environ",
                {"PRIME_API_KEY": "env-key", "PRIME_TEAM_ID": "env-team"},
                clear=False,
            ):
                config = load_prime_config(path)
        self.assertEqual(config.api_key, "env-key")
        self.assertEqual(config.team_id, "env-team")


class PrimeOfferTests(unittest.TestCase):
    def test_parse_offer_normalizes_documented_availability_shape(self) -> None:
        offer = parse_prime_offer(_offer_payload())
        self.assertEqual(len(offer.id), 6)
        self.assertEqual(offer.gpu_type, "H100_80GB")
        self.assertEqual(offer.gpu_memory_gb, 80.0)
        self.assertEqual(offer.price_per_hour, 1.9)
        self.assertEqual(offer.data_center, "CANADA-1")

    def test_offer_memory_uses_per_gpu_size_when_payload_is_aggregate(self) -> None:
        offer = parse_prime_offer(
            _offer_payload(gpuType="A100_80GB", gpuCount=8, gpuMemory=640)
        )
        self.assertEqual(offer.gpu_memory_gb, 640.0)
        self.assertEqual(prime_offer_gpu_memory_gb(offer), 80.0)

    def test_automatic_selection_uses_cheapest_fixed_secure_on_demand_offer(self) -> None:
        rows = [
            parse_prime_offer(_offer_payload(cloudId="expensive", prices={"onDemand": 2.2})),
            parse_prime_offer(_offer_payload(cloudId="spot", prices={"onDemand": 0.2}, isSpot=True)),
            parse_prime_offer(
                _offer_payload(cloudId="variable", prices={"onDemand": 0.5, "isVariable": True})
            ),
            parse_prime_offer(_offer_payload(cloudId="cheap", prices={"onDemand": 1.1})),
        ]
        selected = select_prime_offer(rows, gpu_type="H100_80GB", gpu_count=1)
        self.assertEqual(selected.cloud_id, "cheap")

    def test_exact_offer_selection_can_choose_an_explicit_offer(self) -> None:
        row = parse_prime_offer(_offer_payload())
        self.assertIs(select_prime_offer([row], offer_id=row.id), row)

    def test_out_of_stock_offer_is_not_deployable(self) -> None:
        row = parse_prime_offer(_offer_payload(stockStatus="out_of_stock"))
        available = parse_prime_offer(_offer_payload(cloudId="available"))

        self.assertFalse(is_compatible_prime_offer(row))
        with self.assertRaisesRegex(ValueError, "unavailable"):
            select_prime_offer([row], offer_id=row.id)
        self.assertIs(select_prime_offer([row, available]), available)

    def test_automatic_selection_excludes_cpu_nodes(self) -> None:
        cpu = parse_prime_offer(
            _offer_payload(
                cloudId="cpu",
                gpuType="CPU_NODE",
                gpuMemory=0,
                prices={"onDemand": 0.05},
            )
        )
        gpu = parse_prime_offer(
            _offer_payload(
                cloudId="gpu",
                gpuType="H100_80GB",
                gpuMemory=80,
                prices={"onDemand": 2.0},
            )
        )

        selected = select_prime_offer([cpu, gpu])

        self.assertIs(selected, gpu)

    def test_exact_cpu_offer_is_rejected_for_inference(self) -> None:
        cpu = parse_prime_offer(
            _offer_payload(
                cloudId="cpu",
                gpuType="CPU_NODE",
                gpuMemory=0,
            )
        )

        with self.assertRaisesRegex(ValueError, "not a GPU instance"):
            select_prime_offer([cpu], offer_id=cpu.id)

    def test_selection_applies_model_vram_requirement_with_headroom(self) -> None:
        one_h100 = parse_prime_offer(
            _offer_payload(cloudId="one", gpuCount=1, gpuMemory=80)
        )
        two_h100 = parse_prime_offer(
            _offer_payload(cloudId="two", gpuCount=2, gpuMemory=80)
        )

        selected = select_prime_offer(
            [one_h100, two_h100],
            required_vram_gb=77.0,
        )

        self.assertIs(selected, two_h100)

    def test_inference_excludes_offer_without_bootstrap_image(self) -> None:
        offer = parse_prime_offer(_offer_payload(images=["cuda_12_4_pytorch_2_5"]))

        self.assertFalse(is_compatible_prime_offer(offer, required_vram_gb=8.0))

    def test_automatic_selection_skips_offer_without_bootstrap_image(self) -> None:
        unsupported = parse_prime_offer(
            _offer_payload(
                cloudId="cheap",
                images=["cuda_12_4_pytorch_2_5"],
                prices={"onDemand": 0.5},
            )
        )
        supported = parse_prime_offer(
            _offer_payload(cloudId="supported", prices={"onDemand": 1.5})
        )

        selected = select_prime_offer([unsupported, supported])

        self.assertIs(selected, supported)

    def test_exact_offer_without_bootstrap_image_is_rejected(self) -> None:
        offer = parse_prime_offer(_offer_payload(images=["cuda_12_4_pytorch_2_5"]))

        with self.assertRaisesRegex(ValueError, "does not support runtime image ubuntu_22_cuda_12"):
            select_prime_offer([offer], offer_id=offer.id)

    def test_portable_strategy_accepts_default_cuda_image_offers(self) -> None:
        offer = parse_prime_offer(_offer_payload(images=["ubuntu_22_cuda_12"]))

        self.assertTrue(
            is_compatible_prime_offer(
                offer,
                required_vram_gb=70.0,
                required_image=PRIME_DEFAULT_BOOTSTRAP_IMAGE,
            )
        )
        self.assertIs(
            select_prime_offer(
                [offer],
                offer_id=offer.id,
                required_image=PRIME_DEFAULT_BOOTSTRAP_IMAGE,
            ),
            offer,
        )


class PrimeLaunchSpecTests(unittest.TestCase):
    def test_runtime_uses_prime_default_image_and_upstream_container(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            provider=ComputeProvider.PRIME,
            repo_id="unsloth/Qwen3.8-27B-GGUF",
            provider_options=PrimeProviderOptions(),
        )
        launch = resolve_prime_launch_spec(config)
        preferred_image = preferred_prime_offer_image(BackendType.LLAMACPP)

        self.assertEqual(launch.offer_image, PRIME_DEFAULT_BOOTSTRAP_IMAGE)
        self.assertEqual(
            launch.container_image,
            default_prime_container_image(BackendType.LLAMACPP),
        )
        self.assertEqual(
            preferred_image,
            PRIME_DEFAULT_BOOTSTRAP_IMAGE,
        )


class _FakeResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self._payload = payload
        self.content = b"{}"
        self.text = ""

    def json(self) -> object:
        return self._payload


class _FakeSession:
    def __init__(self, responses: list[_FakeResponse]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def request(self, method: str, url: str, **kwargs: object) -> _FakeResponse:
        self.calls.append({"method": method, "url": url, **kwargs})
        return self.responses.pop(0)


class PrimeBackendTests(unittest.TestCase):
    def test_request_formats_structured_422_detail_without_echoing_input(self) -> None:
        session = _FakeSession(
            [
                _FakeResponse(
                    422,
                    {
                        "detail": [
                            {
                                "type": "literal_error",
                                "loc": ["body", "pod", "image"],
                                "msg": "Input should be a supported image",
                                "input": "endpoint-secret",
                            }
                        ]
                    },
                )
            ]
        )
        backend = PrimeBackend(PrimeConfig(api_key="secret"), session=session)  # type: ignore[arg-type]

        with self.assertRaises(PrimeApiError) as raised:
            backend._request("POST", "/pods", json_payload={})

        self.assertIn(
            "body.pod.image: Input should be a supported image",
            str(raised.exception),
        )
        self.assertNotIn("endpoint-secret", str(raised.exception))

    def test_request_reads_error_field_used_by_prime(self) -> None:
        session = _FakeSession(
            [_FakeResponse(422, {"error": {"message": "Image is not compatible"}})]
        )
        backend = PrimeBackend(PrimeConfig(api_key="secret"), session=session)  # type: ignore[arg-type]

        with self.assertRaisesRegex(PrimeApiError, "Image is not compatible"):
            backend._request("POST", "/pods", json_payload={})

    def test_list_offers_uses_documented_paginated_gpu_endpoint(self) -> None:
        session = _FakeSession([_FakeResponse(200, {"items": [_offer_payload()], "totalCount": 1})])
        backend = PrimeBackend(PrimeConfig(api_key="secret"), session=session)  # type: ignore[arg-type]
        offers = backend.list_offers(gpu_type="H100_80GB", gpu_count=1, region="canada")
        self.assertEqual(len(offers), 1)
        self.assertTrue(str(session.calls[0]["url"]).endswith("/api/v1/availability/gpus"))
        self.assertEqual(session.calls[0]["params"], {
            "gpu_type": "H100_80GB",
            "gpu_count": 1,
            "page": 1,
            "page_size": 100,
        })

    def test_list_offers_filters_country_locally_without_sending_invalid_region(self) -> None:
        session = _FakeSession(
            [
                _FakeResponse(
                    200,
                    {
                        "items": [
                            _offer_payload(cloudId="ca", country="CA", region="north_america"),
                            _offer_payload(cloudId="us", country="US", region="north_america"),
                        ],
                        "totalCount": 2,
                    },
                )
            ]
        )
        backend = PrimeBackend(PrimeConfig(api_key="secret"), session=session)  # type: ignore[arg-type]

        offers = backend.list_offers(region="CA")

        self.assertEqual([offer.country for offer in offers], ["CA"])
        params = session.calls[0]["params"]
        assert isinstance(params, dict)
        self.assertNotIn("regions", params)

    def test_list_offers_hides_cpu_rows_returned_by_gpu_endpoint(self) -> None:
        session = _FakeSession(
            [
                _FakeResponse(
                    200,
                    {
                        "items": [
                            _offer_payload(
                                cloudId="cpu",
                                gpuType="CPU_NODE",
                                gpuMemory=0,
                            ),
                            _offer_payload(cloudId="gpu"),
                        ],
                        "totalCount": 2,
                    },
                )
            ]
        )
        backend = PrimeBackend(PrimeConfig(api_key="secret"), session=session)  # type: ignore[arg-type]

        offers = backend.list_offers()

        self.assertEqual([offer.cloud_id for offer in offers], ["gpu"])

    def test_disk_lifecycle_uses_documented_endpoints_and_team(self) -> None:
        disk_offer_payload = {
            "cloudId": "disk-cloud",
            "provider": "runpod",
            "dataCenter": "US-CA-2",
            "country": "US",
            "region": "united_states",
            "stockStatus": "Available",
            "spec": {
                "minCount": 20,
                "maxCount": 1000,
                "pricePerUnit": 0.0001,
            },
        }
        session = _FakeSession(
            [
                _FakeResponse(200, {"items": [disk_offer_payload], "totalCount": 1}),
                _FakeResponse(200, {"id": "disk-1", "priceHr": 0.003}),
                _FakeResponse(200, {"id": "disk-1", "status": "UNATTACHED"}),
                _FakeResponse(200, {"status": "TERMINATED"}),
            ]
        )
        backend = PrimeBackend(
            PrimeConfig(api_key="secret", team_id="team-1"),
            session=session,  # type: ignore[arg-type]
        )

        offers = backend.list_disk_offers()
        created = backend.create_disk(offers[0], size_gb=30, name="llp-e2e-disk")
        fetched = backend.get_disk("disk-1")
        backend.delete_disk("disk-1")

        self.assertEqual(offers[0].minimum_size_gb, 20)
        self.assertEqual(offers[0].price_per_gb_hour, 0.0001)
        self.assertEqual(created["id"], "disk-1")
        self.assertEqual(fetched["status"], "UNATTACHED")
        create_body = session.calls[1]["json"]
        assert isinstance(create_body, dict)
        self.assertEqual(create_body["team"], {"teamId": "team-1"})
        self.assertEqual(create_body["provider"], {"type": "runpod"})
        self.assertEqual(create_body["disk"]["size"], 30)  # type: ignore[index]
        self.assertTrue(str(session.calls[3]["url"]).endswith("/api/v1/disks/disk-1"))

    def test_create_disk_validates_availability_size_bounds(self) -> None:
        backend = PrimeBackend(PrimeConfig(api_key="secret"), session=_FakeSession([]))  # type: ignore[arg-type]
        offer = PrimeDiskOffer(
            cloud_id="cloud",
            provider_name="runpod",
            data_center="US-1",
            country="US",
            region="united_states",
            stock_status="Available",
            price_per_gb_hour=0.0001,
            minimum_size_gb=50,
            maximum_size_gb=100,
            raw={},
        )

        with self.assertRaisesRegex(ValueError, "at least 50"):
            backend.create_disk(offer, size_gb=30, name="too-small")

    def test_create_tunnel_uses_team_and_pod_labels(self) -> None:
        session = _FakeSession(
            [
                _FakeResponse(
                    201,
                    {
                        "tunnel_id": "t-0-abc",
                        "hostname": "t-0-abc.tunnel.pinfra.io",
                        "url": "https://t-0-abc.tunnel.pinfra.io",
                        "frp_token": "frp-secret",
                        "binding_secret": "binding-secret",
                        "server_host": "tunnel.pinfra.io",
                        "server_port": 7000,
                        "expires_at": "2026-08-25T00:00:00Z",
                        "labels": ["llm-launchpad", "pod:pod-1"],
                    },
                )
            ]
        )
        backend = PrimeBackend(
            PrimeConfig(api_key="secret", team_id="team-1"),
            session=session,  # type: ignore[arg-type]
        )

        tunnel = backend.create_tunnel(
            "pod-1",
            name="llp-prime-vllm-qwen",
        )

        self.assertEqual(tunnel.tunnel_id, "t-0-abc")
        self.assertEqual(tunnel.url, "https://t-0-abc.tunnel.pinfra.io")
        self.assertEqual(
            session.calls[0]["json"],
            {
                "name": "llp-prime-vllm-qwen",
                "local_port": 8000,
                "labels": ["llm-launchpad", "pod:pod-1"],
                "teamId": "team-1",
            },
        )

    def test_delete_pod_removes_associated_tunnel_first(self) -> None:
        session = _FakeSession(
            [
                _FakeResponse(
                    200,
                    {
                        "total": 1,
                        "tunnels": [
                            {
                                "tunnel_id": "t-0-abc",
                                "hostname": "t-0-abc.tunnel.pinfra.io",
                                "url": "https://t-0-abc.tunnel.pinfra.io",
                                "expires_at": "2026-08-25T00:00:00Z",
                                "status": "connected",
                                "labels": ["llm-launchpad", "pod:pod-1"],
                            }
                        ],
                    },
                ),
                _FakeResponse(200, {"success": True}),
                _FakeResponse(200, {"status": "TERMINATED"}),
            ]
        )
        backend = PrimeBackend(PrimeConfig(api_key="secret"), session=session)  # type: ignore[arg-type]

        backend.delete_pod("pod-1")

        self.assertTrue(str(session.calls[0]["url"]).endswith("/api/v1/tunnel"))
        self.assertTrue(str(session.calls[1]["url"]).endswith("/api/v1/tunnel/t-0-abc"))
        self.assertTrue(str(session.calls[2]["url"]).endswith("/api/v1/pods/pod-1"))

    def test_create_vllm_pod_sends_bootstrap_key_team_and_disk(self) -> None:
        session = _FakeSession([_FakeResponse(200, {"id": "pod-1"})])
        backend = PrimeBackend(
            PrimeConfig(api_key="secret", team_id="team-1"), session=session  # type: ignore[arg-type]
        )
        config = DeploymentConfig(
            backend=BackendType.VLLM,
            provider=ComputeProvider.PRIME,
            app_name="llp-prime-vllm-qwen",
            model_name="Qwen/Qwen3-4B",
            served_model_name="qwen",
            endpoint_api_key="endpoint-secret",
            provider_options=PrimeProviderOptions(disk_id="disk-1"),
        )
        with patch.object(
            backend,
            "ensure_bootstrap_ssh_key",
            return_value="ssh-key-1",
        ):
            backend.create_pod(
                config,
                parse_prime_offer(_offer_payload(images=["ubuntu_22_cuda_12"])),
            )
        body = session.calls[0]["json"]
        assert isinstance(body, dict)
        self.assertEqual(body["disks"], ["disk-1"])
        self.assertEqual(body["team"], {"teamId": "team-1"})
        pod = body["pod"]
        assert isinstance(pod, dict)
        self.assertEqual(pod["image"], "ubuntu_22_cuda_12")
        self.assertEqual(pod["sshKeyId"], "ssh-key-1")
        self.assertNotIn("customTemplateId", pod)
        self.assertNotIn("envVars", pod)
        self.assertNotIn("endpoint-secret", str(body))

    def test_create_llamacpp_pod_uses_default_image_and_dedicated_ssh_key(self) -> None:
        session = _FakeSession([_FakeResponse(200, {"id": "pod-portable"})])
        backend = PrimeBackend(
            PrimeConfig(api_key="secret", team_id="team-1"), session=session  # type: ignore[arg-type]
        )
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            provider=ComputeProvider.PRIME,
            app_name="llp-prime-llamacpp-qwen",
            repo_id="unsloth/Qwen3.8-27B-GGUF",
            quant="UD-Q2_K_XL",
            endpoint_api_key="endpoint-secret",
            provider_options=PrimeProviderOptions(),
        )

        with patch.object(
            backend,
            "ensure_bootstrap_ssh_key",
            return_value="ssh-key-1",
        ) as ensure_key:
            backend.create_pod(
                config,
                parse_prime_offer(_offer_payload(images=["ubuntu_22_cuda_12"])),
            )

        ensure_key.assert_called_once_with()
        body = session.calls[0]["json"]
        assert isinstance(body, dict)
        pod = body["pod"]
        assert isinstance(pod, dict)
        self.assertEqual(pod["image"], "ubuntu_22_cuda_12")
        self.assertEqual(pod["sshKeyId"], "ssh-key-1")
        self.assertNotIn("customTemplateId", pod)
        self.assertNotIn("envVars", pod)
        self.assertNotIn("endpoint-secret", str(body))

    def test_bootstrap_ssh_key_reuses_matching_prime_key(self) -> None:
        public_key = "ssh-ed25519 AAAATEST local-comment"
        session = _FakeSession(
            [
                _FakeResponse(
                    200,
                    {
                        "data": [
                            {
                                "id": "existing-key",
                                "publicKey": "ssh-ed25519 AAAATEST remote-comment",
                            }
                        ]
                    },
                )
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            private_path = Path(tmp) / "bootstrap_ed25519"
            private_path.write_text("private", encoding="utf-8")
            Path(f"{private_path}.pub").write_text(public_key, encoding="utf-8")
            backend = PrimeBackend(
                PrimeConfig(api_key="secret"),
                session=session,  # type: ignore[arg-type]
                ssh_private_key_path=private_path,
            )

            key_id = backend.ensure_bootstrap_ssh_key()

        self.assertEqual(key_id, "existing-key")
        self.assertEqual(len(session.calls), 1)

    def test_bootstrap_ssh_key_registers_dedicated_key_when_missing(self) -> None:
        public_key = "ssh-ed25519 AAAATEST llm-launchpad-bootstrap"
        session = _FakeSession(
            [
                _FakeResponse(200, {"data": []}),
                _FakeResponse(200, {"id": "created-key"}),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            private_path = Path(tmp) / "bootstrap_ed25519"
            private_path.write_text("private", encoding="utf-8")
            Path(f"{private_path}.pub").write_text(public_key, encoding="utf-8")
            backend = PrimeBackend(
                PrimeConfig(api_key="secret"),
                session=session,  # type: ignore[arg-type]
                ssh_private_key_path=private_path,
            )

            key_id = backend.ensure_bootstrap_ssh_key()

        self.assertEqual(key_id, "created-key")
        self.assertEqual(session.calls[1]["method"], "POST")
        self.assertEqual(
            session.calls[1]["json"],
            {
                "name": "llm-launchpad-bootstrap",
                "publicKey": public_key,
            },
        )

    def test_portable_bootstrap_keeps_secrets_out_of_uploaded_script(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.VLLM,
            provider=ComputeProvider.PRIME,
            model_name="Qwen/Qwen3-4B",
            endpoint_api_key="endpoint-secret",
            provider_options=PrimeProviderOptions(),
        )
        with patch("huggingface_hub.get_token", return_value="hf-secret"):
            launch = resolve_prime_launch_spec(config)
            env_file = PrimeBackend._docker_env_file(config)
            script = PrimeBackend._bootstrap_script(config, launch)

        self.assertIn("VLLM_API_KEY=endpoint-secret", env_file)
        self.assertIn("HF_TOKEN=hf-secret", env_file)
        self.assertNotIn("endpoint-secret", script)
        self.assertNotIn("hf-secret", script)
        self.assertIn('$VLLM_API_KEY', script)
        self.assertIn(launch.container_image, script)
        self.assertIn("127.0.0.1:8000:8000", script)
        self.assertIn("Waiting for the Prime image to finish installing Docker", script)
        self.assertIn("after 180 seconds", script)

    def test_direct_http_fallback_publishes_runtime_on_all_interfaces(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.VLLM,
            provider=ComputeProvider.PRIME,
            model_name="Qwen/Qwen3-4B",
            endpoint_api_key="endpoint-secret",
            provider_options=PrimeProviderOptions(allow_insecure_http=True),
        )
        launch = resolve_prime_launch_spec(config)

        command = PrimeBackend._bootstrap_docker_command(config, launch)

        self.assertIn("8000:8000", command)
        self.assertNotIn("127.0.0.1:8000:8000", command)

    def test_privileged_command_supports_root_and_passwordless_sudo_images(self) -> None:
        command = PrimeBackend._privileged_shell_command(
            "install -d -m 700 /opt/llm-launchpad"
        )

        self.assertIn('if [ "$(id -u)" -eq 0 ]', command)
        self.assertIn("sh -c 'install -d -m 700 /opt/llm-launchpad'", command)
        self.assertIn("sudo -n sh -c", command)

    def test_tunnel_config_matches_prime_frpc_protocol_and_keeps_api_key_out(self) -> None:
        tunnel = PrimeTunnel(
            tunnel_id="t-0-abc",
            hostname="t-0-abc.tunnel.pinfra.io",
            url="https://t-0-abc.tunnel.pinfra.io",
            frp_token="frp-secret",
            binding_secret="binding-secret",
            server_host="tunnel.pinfra.io",
            server_port=7000,
        )

        config = PrimeBackend._tunnel_config(tunnel)
        script = PrimeBackend._tunnel_start_script()

        self.assertIn('auth.token = "frp-secret"', config)
        self.assertIn('metadatas.binding_secret = "binding-secret"', config)
        self.assertIn('localIP = "127.0.0.1"', config)
        self.assertIn("localPort = 8000", config)
        self.assertIn("/opt/llm-launchpad/frpc", script)
        self.assertNotIn("github.com", script)
        self.assertNotIn("frp-secret", script)
        self.assertNotIn("binding-secret", script)

    def test_start_tunnel_uses_cached_client_instead_of_github(self) -> None:
        backend = PrimeBackend(PrimeConfig(api_key="secret"))
        tunnel = PrimeTunnel(
            tunnel_id="t-0-abc",
            hostname="t-0-abc.tunnel.pinfra.io",
            url="https://t-0-abc.tunnel.pinfra.io",
            frp_token="frp-secret",
            binding_secret="binding-secret",
            server_host="tunnel.pinfra.io",
            server_port=7000,
        )
        uploaded: list[str] = []

        def write_file(
            _pod: dict[str, object],
            filename: str,
            _content: str,
            *,
            mode: str,
        ) -> None:
            uploaded.append(filename)

        with (
            patch.object(backend, "_write_remote_runtime_file", side_effect=write_file),
            patch.object(backend, "_ensure_remote_frpc") as ensure_frpc,
            patch.object(backend, "_pod_frpc_arch", return_value="amd64"),
            patch.object(
                backend,
                "_run_privileged_ssh",
                return_value=subprocess.CompletedProcess([], 0, "", ""),
            ),
        ):
            backend.start_tunnel({"id": "pod-1"}, tunnel)

        ensure_frpc.assert_called_once()
        self.assertEqual(uploaded, ["tunnel.toml", "tunnel-bootstrap.sh"])
        self.assertNotIn("github.com", PrimeBackend._tunnel_start_script())

    def test_tunnel_runtime_status_accepts_connected_prime_status(self) -> None:
        backend = PrimeBackend(PrimeConfig(api_key="secret"))
        running = subprocess.CompletedProcess([], 0, "", "")
        tunnel = PrimeTunnel(
            tunnel_id="t-0-abc",
            hostname="t-0-abc.tunnel.pinfra.io",
            url="https://t-0-abc.tunnel.pinfra.io",
            status="connected",
        )

        with (
            patch.object(backend, "_run_ssh", return_value=running),
            patch.object(backend, "tunnel_runtime_logs", return_value=[]),
            patch.object(backend, "get_tunnel", return_value=tunnel),
        ):
            ready, failed, detail = backend.tunnel_runtime_status(
                {"id": "pod-1"},
                "t-0-abc",
            )

        self.assertTrue(ready)
        self.assertFalse(failed)
        self.assertIn("HTTPS tunnel is connected", detail)

    def test_llamacpp_portable_bootstrap_passes_api_key_without_embedding_secret(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            provider=ComputeProvider.PRIME,
            repo_id="unsloth/Qwen3.8-27B-GGUF",
            quant="UD-Q2_K_XL",
            served_model_name="qwen38-27b",
            endpoint_api_key="endpoint-secret",
            provider_options=PrimeProviderOptions(),
        )
        launch = resolve_prime_launch_spec(config)
        env_file = PrimeBackend._docker_env_file(config)
        script = PrimeBackend._bootstrap_script(config, launch)

        self.assertIn("LLAMA_ARG_API_KEY=endpoint-secret", env_file)
        self.assertNotIn("endpoint-secret", script)
        self.assertIn('--api-key "$LLAMA_ARG_API_KEY"', script)
        self.assertIn("--entrypoint /bin/sh", script)

    def test_llamacpp_portable_disk_uses_only_persistent_cache(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            provider=ComputeProvider.PRIME,
            repo_id="bartowski/Qwen2.5-0.5B-Instruct-GGUF",
            quant="Q4_K_M",
            endpoint_api_key="endpoint-secret",
            provider_options=PrimeProviderOptions(disk_id="disk-1"),
        )
        launch = resolve_prime_launch_spec(config)
        env_file = PrimeBackend._docker_env_file(config)
        command = PrimeBackend._bootstrap_docker_command(config, launch)

        self.assertIn("LLAMA_CACHE=/data/llama.cpp", env_file)
        self.assertIn("/data:/data", command)
        self.assertNotIn(
            "llm-launchpad-llama-cache:/root/.cache/llama.cpp",
            command,
        )

    def test_llamacpp_portable_bootstrap_leaves_gpu_layers_unset_for_auto_fit(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            provider=ComputeProvider.PRIME,
            repo_id="unsloth/Qwen3.8-Flash-Next-GGUF",
            quant="UD-Q2_K_XL",
            endpoint_api_key="endpoint-secret",
            provider_options=PrimeProviderOptions(),
        )
        launch = resolve_prime_launch_spec(config)

        env = PrimeBackend.runtime_env(config)
        command = PrimeBackend._bootstrap_docker_command(config, launch)

        self.assertNotIn("N_GPU_LAYERS", env)
        self.assertNotIn("--n-gpu-layers", command[-1])

    def test_llamacpp_portable_bootstrap_preserves_gpu_layer_override(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            provider=ComputeProvider.PRIME,
            repo_id="unsloth/Qwen3.8-Flash-Next-GGUF",
            quant="UD-Q2_K_XL",
            endpoint_api_key="endpoint-secret",
            n_gpu_layers=42,
            provider_options=PrimeProviderOptions(),
        )
        launch = resolve_prime_launch_spec(config)

        command = PrimeBackend._bootstrap_docker_command(config, launch)

        self.assertIn("--n-gpu-layers 42", command[-1])

    def test_portable_bootstrap_rejects_missing_endpoint_key(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            provider=ComputeProvider.PRIME,
            repo_id="unsloth/Qwen3.8-27B-GGUF",
            provider_options=PrimeProviderOptions(),
        )

        with self.assertRaisesRegex(ValueError, "require an endpoint API key"):
            PrimeBackend._docker_env_file(config)

    def test_portable_runtime_status_detects_restart_loop(self) -> None:
        backend = PrimeBackend(PrimeConfig(api_key="secret"))
        probe = subprocess.CompletedProcess([], 22, "", "not ready")
        state = subprocess.CompletedProcess([], 0, "running|2|3\n", "")

        with (
            patch.object(backend, "_run_ssh", side_effect=[probe, state]),
            patch.object(
                backend,
                "bootstrap_runtime_logs",
                return_value=["vllm: error: unsupported option"],
            ),
        ):
            ready, failed, detail = backend.bootstrap_runtime_status({"id": "pod-1"})

        self.assertFalse(ready)
        self.assertTrue(failed)
        self.assertIn("restart loop (count 3)", detail)
        self.assertIn("unsupported option", detail)

    def test_portable_runtime_status_reports_running_model_load(self) -> None:
        backend = PrimeBackend(PrimeConfig(api_key="secret"))
        probe = subprocess.CompletedProcess([], 22, "", "not ready")
        state = subprocess.CompletedProcess([], 0, "running|0|0\n", "")
        stats = subprocess.CompletedProcess([], 0, "1.2GB / 8GB\n", "")

        with patch.object(backend, "_run_ssh", side_effect=[probe, state, stats]):
            ready, failed, detail = backend.bootstrap_runtime_status({"id": "pod-1"})

        self.assertFalse(ready)
        self.assertFalse(failed)
        self.assertEqual(
            detail,
            "runtime container is loading the model (network 1.2GB / 8GB)",
        )

    def test_portable_runtime_status_reports_download_percent(self) -> None:
        backend = PrimeBackend(PrimeConfig(api_key="secret"))
        probe = subprocess.CompletedProcess([], 22, "", "not ready")
        state = subprocess.CompletedProcess([], 0, "running|0|0\n", "")
        stats = subprocess.CompletedProcess(
            [],
            0,
            "61.7GB / 551MB\nCACHE:59000000000\nEXPECTED:78900000000\n",
            "",
        )

        with patch.object(backend, "_run_ssh", side_effect=[probe, state, stats]):
            ready, failed, detail = backend.bootstrap_runtime_status({"id": "pod-1"})

        self.assertFalse(ready)
        self.assertFalse(failed)
        self.assertEqual(detail, "runtime container is downloading the model (74%)")

    def test_runtime_load_detail_helpers_parse_progress(self) -> None:
        self.assertEqual(PrimeBackend._expected_model_bytes(
            DeploymentConfig(required_vram_gb=78.9)
        ), 78_900_000_000)
        self.assertEqual(PrimeBackend._parse_docker_byte_count("61.7GB"), 61_700_000_000)
        network, cache_bytes, expected_bytes = PrimeBackend._parse_runtime_progress_output(
            "61.7GB / 551MB\nCACHE:59000000000\nEXPECTED:78900000000\n"
        )
        self.assertEqual(network, "61.7GB / 551MB")
        self.assertEqual(cache_bytes, 59_000_000_000)
        self.assertEqual(expected_bytes, 78_900_000_000)
        self.assertEqual(
            PrimeBackend._runtime_load_detail(
                network=network,
                cache_bytes=cache_bytes,
                expected_bytes=expected_bytes,
            ),
            "runtime container is downloading the model (74%)",
        )

    def test_portable_runtime_status_reports_exit_code_without_logs(self) -> None:
        backend = PrimeBackend(PrimeConfig(api_key="secret"))
        probe = subprocess.CompletedProcess([], 22, "", "not ready")
        state = subprocess.CompletedProcess([], 0, "exited|137|0\n", "")

        with (
            patch.object(backend, "_run_ssh", side_effect=[probe, state]),
            patch.object(backend, "bootstrap_runtime_logs", return_value=[]),
        ):
            ready, failed, detail = backend.bootstrap_runtime_status({"id": "pod-1"})

        self.assertFalse(ready)
        self.assertTrue(failed)
        self.assertEqual(detail, "runtime container is exited (exit code 137)")

    def test_portable_runtime_log_redacts_quoted_api_key_list(self) -> None:
        line = "non-default args: {'api_key': ['endpoint-secret'], 'model': 'test'}"

        redacted = PrimeBackend._redact_runtime_log(line)

        self.assertNotIn("endpoint-secret", redacted)
        self.assertIn("'api_key': [redacted]", redacted)

        tunnel_line = 'metadatas.binding_secret = "binding-secret"'
        tunnel_redacted = PrimeBackend._redact_runtime_log(tunnel_line)
        self.assertNotIn("binding-secret", tunnel_redacted)

    def test_llamacpp_runtime_env_preserves_backend_configuration(self) -> None:
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            provider=ComputeProvider.PRIME,
            app_name="llp-prime-llamacpp-qwen",
            repo_id="Qwen/Qwen3-8B-GGUF",
            quant="Q4_K_M",
            served_model_name="qwen-gguf",
            endpoint_api_key="endpoint-secret",
            n_gpu_layers=99,
            server_args='--ctx-size 65536 --chat-template "chat ml"',
            provider_options=PrimeProviderOptions(disk_id="disk-llama"),
        )

        with patch("huggingface_hub.get_token", return_value="hf-secret"):
            env = PrimeBackend.runtime_env(config)

        self.assertEqual(env["REPO_ID"], "Qwen/Qwen3-8B-GGUF")
        self.assertEqual(env["QUANT"], "Q4_K_M")
        self.assertEqual(env["SERVED_MODEL_NAME"], "qwen-gguf")
        self.assertEqual(env["N_GPU_LAYERS"], "99")
        self.assertEqual(env["LLAMACPP_API_KEY"], "endpoint-secret")
        self.assertEqual(
            env["LLAMACPP_SERVER_ARGS"],
            "--ctx-size\n65536\n--chat-template\nchat ml",
        )
        self.assertEqual(env["LLAMA_CACHE"], "/data/llama.cpp")
        self.assertEqual(env["HF_TOKEN"], "hf-secret")

    def test_endpoint_url_requires_explicit_opt_in_for_http(self) -> None:
        pod = {
            "ip": "203.0.113.10",
            "primePortMapping": [{"internal": "*", "external": "*", "protocol": "TCP"}],
        }
        with self.assertRaisesRegex(ValueError, "HTTP-only"):
            PrimeBackend.endpoint_url(pod)
        self.assertEqual(
            PrimeBackend.endpoint_url(pod, allow_insecure_http=True),
            "http://203.0.113.10:8000",
        )

    def test_default_image_direct_ip_also_requires_explicit_http_opt_in(self) -> None:
        pod = {"ip": ["203.0.113.11"]}

        with self.assertRaisesRegex(ValueError, "explicit HTTP opt-in"):
            PrimeBackend.endpoint_url(pod, allow_direct_ip=True)
        self.assertEqual(
            PrimeBackend.endpoint_url(
                pod,
                allow_direct_ip=True,
                allow_insecure_http=True,
            ),
            "http://203.0.113.11:8000",
        )


class PrimeBillingWalletTests(unittest.TestCase):
    def test_billing_wallet_reads_documented_wallet_endpoint(self) -> None:
        session = _FakeSession(
            [
                _FakeResponse(
                    200,
                    {
                        "wallet_id": "wallet-1",
                        "team_id": None,
                        "balance_usd": 41.25,
                        "currency": "USD",
                        "total_billings": 3,
                        "recent_billings": [
                            {
                                "id": "b-1",
                                "amount_usd": 1.25,
                                "currency": "USD",
                                "resource_type": "compute",
                                "resource_id": "pod-1",
                            }
                        ],
                    },
                )
            ]
        )
        backend = PrimeBackend(PrimeConfig(api_key="secret"), session=session)  # type: ignore[arg-type]

        payload, error = backend.billing_wallet()

        self.assertIsNone(error)
        assert payload is not None
        self.assertEqual(payload["balance_usd"], 41.25)
        self.assertTrue(str(session.calls[0]["url"]).endswith("/api/v1/billing/wallet"))
        self.assertEqual(session.calls[0]["params"], {"limit": 100})

    def test_billing_wallet_scopes_team_wallet_and_limit_when_configured(self) -> None:
        session = _FakeSession([_FakeResponse(200, {"balance_usd": 0})])
        backend = PrimeBackend(  # type: ignore[arg-type]
            PrimeConfig(api_key="secret", team_id="team-1"), session=session
        )

        payload, error = backend.billing_wallet(limit=10)

        self.assertIsNone(error)
        assert payload is not None
        self.assertEqual(payload["balance_usd"], 0)
        self.assertEqual(session.calls[0]["params"], {"limit": 10, "teamId": "team-1"})

    def test_billing_wallet_returns_error_tuple_on_api_failure(self) -> None:
        session = _FakeSession([_FakeResponse(401, {})])
        backend = PrimeBackend(PrimeConfig(api_key="secret"), session=session)  # type: ignore[arg-type]

        payload, error = backend.billing_wallet()

        self.assertIsNone(payload)
        assert error is not None
        self.assertIn("authentication failed", error)

    def test_billing_wallet_rejects_non_object_payload(self) -> None:
        session = _FakeSession([_FakeResponse(200, ["unexpected"])])
        backend = PrimeBackend(PrimeConfig(api_key="secret"), session=session)  # type: ignore[arg-type]

        payload, error = backend.billing_wallet()

        self.assertIsNone(payload)
        assert error is not None
        self.assertIn("unsupported response", error)


class _OrchestratorPrimeBackend:
    def __init__(self) -> None:
        self.config = PrimeConfig(api_key="secret", user_id="user-1")
        self.offer = parse_prime_offer(_offer_payload())
        self.deleted: list[str] = []

    def preflight(self) -> tuple[bool, str]:
        return True, ""

    def list_offers(self, **_kwargs: object) -> list[ComputeOffer]:
        return [self.offer]

    def list_disk_offers(self) -> list[object]:
        return []

    def create_pod(self, _config: DeploymentConfig, _offer: ComputeOffer) -> dict[str, object]:
        return {"id": "pod-1", "status": "PROVISIONING"}

    def get_pod(self, _pod_id: str) -> dict[str, object]:
        return {"id": "pod-1", "status": "ACTIVE", "installationStatus": "FINISHED"}

    def get_pod_logs(self, _pod_id: str, tail: int = 200) -> list[str]:
        return ["runtime log"]

    def delete_pod(self, pod_id: str) -> None:
        self.deleted.append(pod_id)


class _PortableOrchestratorPrimeBackend(_OrchestratorPrimeBackend):
    def __init__(self, *, bootstrap_failure: bool = False) -> None:
        super().__init__()
        self.bootstrap_failure = bootstrap_failure
        self.bootstrap_started = 0
        self.tunnel_started = 0
        self.endpoint_kwargs: dict[str, object] = {}

    def get_pod(self, _pod_id: str) -> dict[str, object]:
        return {
            "id": "pod-1",
            "status": "ACTIVE",
            "installationStatus": "FINISHED",
            "sshConnection": "ssh ubuntu@203.0.113.12",
            "ip": "203.0.113.12",
        }

    def endpoint_url(
        self,
        _pod: dict[str, object],
        **kwargs: object,
    ) -> str:
        self.endpoint_kwargs = kwargs
        return "http://203.0.113.12:8000"

    def start_bootstrap_runtime(
        self,
        _config: DeploymentConfig,
        _pod: dict[str, object],
    ) -> None:
        self.bootstrap_started += 1

    def bootstrap_runtime_status(
        self,
        _pod: dict[str, object],
    ) -> tuple[bool, bool, str]:
        if self.bootstrap_failure:
            return False, True, "model server exited"
        return True, False, "OpenAI-compatible endpoint is ready"

    def public_endpoint_ready(self, _endpoint: str, _api_key: str) -> tuple[bool, str]:
        return True, ""

    def create_tunnel(
        self,
        _pod_id: str,
        *,
        name: str,
        local_port: int = 8000,
    ) -> PrimeTunnel:
        _ = name, local_port
        return PrimeTunnel(
            tunnel_id="t-0-abc",
            hostname="t-0-abc.tunnel.pinfra.io",
            url="https://t-0-abc.tunnel.pinfra.io",
            frp_token="frp-secret",
            binding_secret="binding-secret",
            server_host="tunnel.pinfra.io",
        )

    def start_tunnel(
        self,
        _pod: dict[str, object],
        _tunnel: PrimeTunnel,
    ) -> None:
        self.tunnel_started += 1

    def tunnel_runtime_status(
        self,
        _pod: dict[str, object],
        _tunnel_id: str,
    ) -> tuple[bool, bool, str]:
        return True, False, "secure HTTPS tunnel is connected"


class _DiskOrchestratorPrimeBackend(_PortableOrchestratorPrimeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.created_disks: list[tuple[object, int, str]] = []
        self.create_pod_disk_ids: list[str | None] = []
        self.list_offers_calls: list[object] = []
        self.disk_offer = PrimeDiskOffer(
            cloud_id="n3-H100x1",
            provider_name="hyperstack",
            data_center="CANADA-1",
            country="CA",
            region="canada",
            stock_status="Available",
            price_per_gb_hour=0.0001,
            minimum_size_gb=30,
            maximum_size_gb=1000,
            raw={},
        )

    def list_offers(self, **kwargs: object) -> list[ComputeOffer]:
        self.list_offers_calls.append(kwargs.get("disk_id"))
        return [self.offer]

    def list_disk_offers(self) -> list[PrimeDiskOffer]:
        return [self.disk_offer]

    def create_disk(
        self,
        offer: PrimeDiskOffer,
        *,
        size_gb: int,
        name: str,
    ) -> dict[str, str]:
        self.created_disks.append((offer, size_gb, name))
        return {"id": "disk-auto"}

    def get_disk(self, disk_id: str) -> dict[str, str]:
        return {"id": disk_id, "status": "UNATTACHED"}

    def create_pod(self, config: DeploymentConfig, offer: ComputeOffer) -> dict[str, object]:
        self.create_pod_disk_ids.append(prime_provider_options(config).disk_id)
        return super().create_pod(config, offer)


class PrimeOrchestratorTests(unittest.TestCase):
    def test_prime_deploy_returns_endpoint_and_generated_bearer_key(self) -> None:
        backend = _PortableOrchestratorPrimeBackend()
        config = DeploymentConfig(
            backend=BackendType.VLLM,
            provider=ComputeProvider.PRIME,
            app_name="llp-prime-vllm-qwen",
            model_name="Qwen/Qwen3-4B",
            provider_options=PrimeProviderOptions(),
        )
        events = list(Orchestrator(prime_backend=backend).deploy(config))  # type: ignore[arg-type]
        milestones = [
            event.line
            for event in events
            if isinstance(event, LogEvent) and event.is_milestone
        ]
        complete = next(
            event
            for event in events
            if isinstance(event, OperationCompleteEvent)
            and event.operation == OperationType.DEPLOY
        )
        self.assertTrue(complete.success)
        self.assertIsInstance(complete.data, EndpointInfo)
        endpoint = complete.data
        assert isinstance(endpoint, EndpointInfo)
        self.assertEqual(endpoint.app_id, "pod-1")
        self.assertEqual(endpoint.provider, ComputeProvider.PRIME)
        self.assertTrue(endpoint.endpoint_api_key)
        self.assertTrue(any(line.startswith("Selected Prime offer ") for line in milestones))
        self.assertIn("Prime pod created: pod-1", milestones)
        self.assertTrue(any(line.startswith("Prime runtime: ") for line in milestones))
        self.assertTrue(any(line.startswith("Prime endpoint: https://") for line in milestones))

    def test_prime_llamacpp_deploy_preserves_gguf_metadata(self) -> None:
        backend = _PortableOrchestratorPrimeBackend()
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            provider=ComputeProvider.PRIME,
            app_name="llp-prime-llamacpp-qwen",
            repo_id="Qwen/Qwen3-8B-GGUF",
            quant="Q4_K_M",
            provider_options=PrimeProviderOptions(),
        )

        events = list(Orchestrator(prime_backend=backend).deploy(config))  # type: ignore[arg-type]
        complete = next(
            event
            for event in events
            if isinstance(event, OperationCompleteEvent)
            and event.operation == OperationType.DEPLOY
        )

        self.assertTrue(complete.success)
        endpoint = complete.data
        self.assertIsInstance(endpoint, EndpointInfo)
        assert isinstance(endpoint, EndpointInfo)
        self.assertEqual(endpoint.backend, BackendType.LLAMACPP)
        self.assertEqual(endpoint.repo_id, "Qwen/Qwen3-8B-GGUF")
        self.assertEqual(endpoint.quant, "Q4_K_M")
        self.assertEqual(endpoint.served_model_name, "Qwen3-8B-GGUF-Q4_K_M")

    def test_prime_portable_deploy_bootstraps_default_image_before_success(self) -> None:
        backend = _PortableOrchestratorPrimeBackend()
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            provider=ComputeProvider.PRIME,
            app_name="llp-prime-llamacpp-qwen",
            repo_id="unsloth/Qwen3.8-27B-GGUF",
            quant="UD-Q2_K_XL",
            provider_options=PrimeProviderOptions(allow_insecure_http=True),
        )
        events = list(Orchestrator(prime_backend=backend).deploy(config))  # type: ignore[arg-type]

        complete = next(event for event in events if isinstance(event, OperationCompleteEvent))
        self.assertTrue(complete.success)
        self.assertEqual(backend.bootstrap_started, 1)
        self.assertEqual(
            backend.endpoint_kwargs,
            {"allow_insecure_http": True, "allow_direct_ip": True},
        )
        self.assertEqual(backend.deleted, [])

    def test_prime_portable_bootstrap_failure_terminates_pod(self) -> None:
        backend = _PortableOrchestratorPrimeBackend(bootstrap_failure=True)
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            provider=ComputeProvider.PRIME,
            app_name="llp-prime-llamacpp-qwen",
            repo_id="unsloth/Qwen3.8-27B-GGUF",
            quant="UD-Q2_K_XL",
            provider_options=PrimeProviderOptions(allow_insecure_http=True),
        )
        events = list(Orchestrator(prime_backend=backend).deploy(config))  # type: ignore[arg-type]

        complete = next(event for event in events if isinstance(event, OperationCompleteEvent))
        self.assertFalse(complete.success)
        self.assertIn("model server exited", complete.detail)
        self.assertEqual(backend.deleted, ["pod-1"])

    def test_prime_portable_deploy_uses_secure_tunnel_without_http_opt_in(self) -> None:
        backend = _PortableOrchestratorPrimeBackend()
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            provider=ComputeProvider.PRIME,
            app_name="llp-prime-llamacpp-qwen",
            repo_id="unsloth/Qwen3.8-27B-GGUF",
            quant="UD-Q2_K_XL",
            provider_options=PrimeProviderOptions(),
        )
        events = list(Orchestrator(prime_backend=backend).deploy(config))  # type: ignore[arg-type]

        complete = next(event for event in events if isinstance(event, OperationCompleteEvent))
        self.assertTrue(complete.success)
        endpoint = complete.data
        self.assertIsInstance(endpoint, EndpointInfo)
        assert isinstance(endpoint, EndpointInfo)
        self.assertEqual(endpoint.web_url, "https://t-0-abc.tunnel.pinfra.io")
        self.assertEqual(backend.bootstrap_started, 1)
        self.assertEqual(backend.tunnel_started, 1)
        self.assertEqual(backend.endpoint_kwargs, {})
        self.assertEqual(backend.deleted, [])
        available = next(
            event for event in events if isinstance(event, EndpointAvailableEvent)
        )
        self.assertEqual(available.endpoint.web_url, "https://t-0-abc.tunnel.pinfra.io")

    def test_prime_tunnel_starts_before_runtime_is_ready(self) -> None:
        class _OverlapBackend(_PortableOrchestratorPrimeBackend):
            def __init__(self) -> None:
                super().__init__()
                self.tunnel_started_during_bootstrap: list[int] = []
                self._polls = 0

            def bootstrap_runtime_status(
                self,
                _pod: dict[str, object],
            ) -> tuple[bool, bool, str]:
                self.tunnel_started_during_bootstrap.append(self.tunnel_started)
                self._polls += 1
                if self._polls < 2:
                    return False, False, "loading the model"
                return True, False, "OpenAI-compatible endpoint is ready"

        backend = _OverlapBackend()
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            provider=ComputeProvider.PRIME,
            app_name="llp-prime-llamacpp-qwen",
            repo_id="unsloth/Qwen3.8-27B-GGUF",
            quant="UD-Q2_K_XL",
            provider_options=PrimeProviderOptions(),
        )
        with patch(
            "llm_launchpad.core.orchestrator.shutdown_event",
            return_value=SimpleNamespace(wait=lambda **_kwargs: False),
        ):
            events = list(Orchestrator(prime_backend=backend).deploy(config))  # type: ignore[arg-type]

        complete = next(event for event in events if isinstance(event, OperationCompleteEvent))
        self.assertTrue(complete.success)
        self.assertIn(1, backend.tunnel_started_during_bootstrap)
        available_index = next(
            index
            for index, event in enumerate(events)
            if isinstance(event, EndpointAvailableEvent)
        )
        complete_index = next(
            index
            for index, event in enumerate(events)
            if isinstance(event, OperationCompleteEvent)
        )
        self.assertLess(available_index, complete_index)

    def test_prime_deploy_creates_and_attaches_cache_disk(self) -> None:
        backend = _DiskOrchestratorPrimeBackend()
        config = DeploymentConfig(
            backend=BackendType.VLLM,
            provider=ComputeProvider.PRIME,
            app_name="llp-prime-vllm-qwen",
            model_name="Qwen/Qwen3-4B",
            provider_options=PrimeProviderOptions(),
        )
        events = list(Orchestrator(prime_backend=backend).deploy(config))  # type: ignore[arg-type]
        complete = next(event for event in events if isinstance(event, OperationCompleteEvent))
        self.assertTrue(complete.success)
        self.assertEqual(backend.created_disks[0][1], 100)
        self.assertEqual(backend.create_pod_disk_ids, ["disk-auto"])
        self.assertTrue(
            any("Created Prime cache disk disk-auto" in event.line for event in events if isinstance(event, LogEvent))
        )

    def test_prime_deploy_reuses_stored_cache_disk(self) -> None:
        backend = _DiskOrchestratorPrimeBackend()
        remember_prime_disk(
            StoredPrimeDisk(
                id="disk-existing",
                provider_name="hyperstack",
                data_center="CANADA-1",
                cloud_id="n3-H100x1",
                size_gb=100,
            )
        )
        config = DeploymentConfig(
            backend=BackendType.VLLM,
            provider=ComputeProvider.PRIME,
            app_name="llp-prime-vllm-qwen",
            model_name="Qwen/Qwen3-4B",
            provider_options=PrimeProviderOptions(),
        )
        events = list(Orchestrator(prime_backend=backend).deploy(config))  # type: ignore[arg-type]
        complete = next(event for event in events if isinstance(event, OperationCompleteEvent))
        self.assertTrue(complete.success)
        self.assertEqual(backend.created_disks, [])
        self.assertEqual(backend.create_pod_disk_ids, ["disk-existing"])
        self.assertIn("disk-existing", backend.list_offers_calls)

    def test_prime_deploy_can_skip_auto_disk(self) -> None:
        backend = _DiskOrchestratorPrimeBackend()
        config = DeploymentConfig(
            backend=BackendType.VLLM,
            provider=ComputeProvider.PRIME,
            app_name="llp-prime-vllm-qwen",
            model_name="Qwen/Qwen3-4B",
            provider_options=PrimeProviderOptions(auto_disk=False),
        )
        events = list(Orchestrator(prime_backend=backend).deploy(config))  # type: ignore[arg-type]
        complete = next(event for event in events if isinstance(event, OperationCompleteEvent))
        self.assertTrue(complete.success)
        self.assertEqual(backend.created_disks, [])
        self.assertEqual(backend.create_pod_disk_ids, [None])

    def test_prime_llamacpp_rejects_non_default_revision(self) -> None:
        backend = _PortableOrchestratorPrimeBackend()
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            provider=ComputeProvider.PRIME,
            app_name="llp-prime-llamacpp-qwen",
            repo_id="Qwen/Qwen3-8B-GGUF",
            quant="Q4_K_M",
            revision="feature-branch",
            provider_options=PrimeProviderOptions(),
        )

        events = list(Orchestrator(prime_backend=backend).deploy(config))  # type: ignore[arg-type]
        complete = next(event for event in events if isinstance(event, OperationCompleteEvent))

        self.assertFalse(complete.success)
        self.assertEqual(complete.exit_code, 2)
        self.assertIn("default Hugging Face revision", complete.detail)

    def test_prime_deploy_rejects_offer_below_model_vram_requirement(self) -> None:
        backend = _PortableOrchestratorPrimeBackend()
        config = DeploymentConfig(
            backend=BackendType.LLAMACPP,
            provider=ComputeProvider.PRIME,
            app_name="llp-prime-llamacpp-large",
            repo_id="org/Large-GGUF",
            quant="Q4_K_M",
            gpu_type="H100_80GB",
            gpu_count=1,
            required_vram_gb=100.0,
            provider_options=PrimeProviderOptions(),
        )

        events = list(Orchestrator(prime_backend=backend).deploy(config))  # type: ignore[arg-type]
        complete = next(event for event in events if isinstance(event, OperationCompleteEvent))

        self.assertFalse(complete.success)
        self.assertIn("No secure on-demand Prime GPU offer", complete.detail)

    def test_prime_deploy_cancellation_terminates_provisioning_pod(self) -> None:
        backend = _PortableOrchestratorPrimeBackend()
        config = DeploymentConfig(
            backend=BackendType.VLLM,
            provider=ComputeProvider.PRIME,
            app_name="llp-prime-vllm-qwen",
            model_name="Qwen/Qwen3-4B",
            provider_options=PrimeProviderOptions(),
        )
        provisioning = {
            "id": "pod-1",
            "status": "PROVISIONING",
            "installationStatus": "RUNNING",
        }
        with (
            patch.object(backend, "get_pod", return_value=provisioning),
            patch(
                "llm_launchpad.core.orchestrator.shutdown_event",
                return_value=SimpleNamespace(wait=lambda **_kwargs: True),
            ),
        ):
            events = list(Orchestrator(prime_backend=backend).deploy(config))  # type: ignore[arg-type]

        complete = next(event for event in events if isinstance(event, OperationCompleteEvent))
        self.assertFalse(complete.success)
        self.assertIn("cancelled during provisioning", complete.detail)
        self.assertEqual(backend.deleted, ["pod-1"])


class ConnectionStoreTests(unittest.TestCase):
    def test_prime_connection_secret_round_trips_with_private_permissions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "connections.json"
            config = DeploymentConfig(
                backend=BackendType.VLLM,
                provider=ComputeProvider.PRIME,
                app_name="llp-prime-vllm-qwen",
                model_name="Qwen/Qwen3-4B",
                endpoint_api_key="endpoint-secret",
            )
            endpoint = EndpointInfo(
                name=config.app_name or "",
                app_id="pod-1",
                backend=BackendType.VLLM,
                provider=ComputeProvider.PRIME,
                web_url="https://prime.example",
            )
            save_connection(config, endpoint, path)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(
                load_connection_entries(path)[endpoint.name]["api_key"], "endpoint-secret"
            )
            row = EndpointInfo(name=endpoint.name, backend=BackendType.VLLM)
            merge_connections([row], path)
            self.assertEqual(row.endpoint_api_key, "endpoint-secret")
            self.assertEqual(row.provider, ComputeProvider.PRIME)
            self.assertEqual(row.app_id, "pod-1")
            remove_connection(endpoint.name, path)
            self.assertEqual(load_connection_entries(path), {})

    def test_rows_from_connection_cache_rebuilds_endpoint_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "connections.json"
            config = DeploymentConfig(
                backend=BackendType.LLAMACPP,
                provider=ComputeProvider.MODAL,
                app_name="llamacpp-qwen",
                model_name="Qwen/Qwen3-4B",
                served_model_name="Qwen3-4B",
                endpoint_api_key="endpoint-secret",
                server_args="--ctx-size 131072",
                max_context_tokens=131072,
                max_output_tokens=32768,
            )
            endpoint = EndpointInfo(
                name=config.app_name or "",
                app_id="app-1",
                backend=BackendType.LLAMACPP,
                provider=ComputeProvider.MODAL,
                web_url="https://alice--llamacpp-qwen.modal.run/v1",
                served_model_name="Qwen3-4B",
            )
            save_connection(config, endpoint, path)
            rows = rows_from_connection_cache(path)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row.name, "llamacpp-qwen")
            self.assertEqual(row.backend, BackendType.LLAMACPP)
            self.assertEqual(row.provider, ComputeProvider.MODAL)
            self.assertEqual(row.web_url, "https://alice--llamacpp-qwen.modal.run")
            self.assertEqual(row.served_model_name, "Qwen3-4B")
            self.assertEqual(row.endpoint_api_key, "endpoint-secret")
            self.assertEqual(row.app_id, "app-1")
            self.assertEqual(row.max_context_tokens, 131072)
            self.assertEqual(row.max_output_tokens, 32768)

    def test_merge_connections_restores_llamacpp_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "connections.json"
            config = DeploymentConfig(
                backend=BackendType.LLAMACPP,
                provider=ComputeProvider.PRIME,
                app_name="llp-prime-llamacpp-qwen",
                repo_id="Qwen/Qwen3-8B-GGUF",
            )
            endpoint = EndpointInfo(
                name=config.app_name or "",
                app_id="pod-llama",
                backend=BackendType.LLAMACPP,
                provider=ComputeProvider.PRIME,
                web_url="https://prime.example",
            )
            save_connection(config, endpoint, path)

            row = EndpointInfo(name=endpoint.name, backend=BackendType.VLLM)
            merge_connections([row], path)

            self.assertEqual(row.backend, BackendType.LLAMACPP)

    def test_reasoning_profile_round_trips_through_connection_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "connections.json"
            config = DeploymentConfig(
                backend=BackendType.LLAMACPP,
                provider=ComputeProvider.MODAL,
                app_name="llamacpp-qwen38",
                repo_id="unsloth/Qwen3.8-27B-GGUF",
                quant="UD-Q4_K_XL",
                reasoning=ReasoningCapabilities(
                    profile_id="hf-test-profile",
                    canonical_model_id="unsloth/Qwen3.8-27B-GGUF",
                    model_revision="a" * 40,
                    efforts=("low", "medium", "xhigh"),
                    default_effort="xhigh",
                    source_repo="Qwen/Qwen3.8-27B",
                    source_revision="b" * 40,
                    source_path="chat_template.jinja",
                    request_option_path="chat_template_kwargs.reasoning_effort",
                ),
            )
            endpoint = EndpointInfo(
                name=config.app_name or "",
                app_id="app-qwen38",
                backend=BackendType.LLAMACPP,
                provider=ComputeProvider.MODAL,
                web_url="https://qwen38.example",
            )

            save_connection(config, endpoint, path)
            entry = load_connection_entries(path)[endpoint.name]
            rows = rows_from_connection_cache(path)

            self.assertEqual(entry["repo_id"], "unsloth/Qwen3.8-27B-GGUF")
            self.assertEqual(
                entry["reasoning"]["efforts"],
                ["low", "medium", "xhigh"],
            )
            self.assertEqual(len(rows), 1)
            self.assertIsNotNone(rows[0].reasoning)
            assert rows[0].reasoning is not None
            self.assertEqual(rows[0].reasoning.default_effort, "xhigh")
