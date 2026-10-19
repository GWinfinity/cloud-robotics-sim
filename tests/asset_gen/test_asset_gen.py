"""Tests for the asset_gen 3D generation API clients (no network needed)."""

from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path

import pytest

from cloud_robotics_sim.asset_gen import (
    AssetGenClient,
    AssetGenError,
    GenerationRequest,
    TaskFailedError,
    TaskHandle,
    TaskTimeoutError,
    configured_providers,
    get_provider,
    hunyuan3d,
    list_providers,
    meshy,
    register_provider,
    rodin,
    tripo,
)
from cloud_robotics_sim.asset_gen import base as base_module
from cloud_robotics_sim.asset_gen import client as client_module
from cloud_robotics_sim.asset_gen.base import (
    download_file,
    poll_task,
    safe_task_id,
    validate_url,
)

# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


class TestGenerationRequest:
    """Request field validation."""

    def test_prompt_or_image_exclusive(self):
        with pytest.raises(ValueError):
            GenerationRequest()
        with pytest.raises(ValueError):
            GenerationRequest(prompt="x", image_path="a.png")
        GenerationRequest(prompt="a chair")  # ok
        with pytest.raises(FileNotFoundError):
            GenerationRequest(image_path="/nonexistent/nope.png")

    def test_topology_validation(self):
        with pytest.raises(ValueError):
            GenerationRequest(prompt="x", topology="ngon")
        GenerationRequest(prompt="x", topology="quad")  # ok

    def test_image_request_ok(self, tmp_path):
        img = tmp_path / "a.png"
        img.write_bytes(b"\x89PNG fake")
        req = GenerationRequest(image_path=img)
        assert not req.is_text


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    """Provider registry and credential detection."""

    def test_builtin_providers_registered(self):
        assert list_providers() == ["hunyuan3d", "meshy", "rodin", "tripo"]

    def test_configured_follows_env(self, monkeypatch):
        monkeypatch.delenv("TRIPO_API_KEY", raising=False)
        assert not get_provider("tripo").configured()
        monkeypatch.setenv("TRIPO_API_KEY", "tsk_test")
        assert get_provider("tripo").configured()

    def test_configured_providers_empty_without_keys(self, monkeypatch):
        for var in (
            "TRIPO_API_KEY",
            "MESHY_API_KEY",
            "RODIN_API_KEY",
            "TENCENTCLOUD_SECRET_ID",
            "TENCENTCLOUD_SECRET_KEY",
        ):
            monkeypatch.delenv(var, raising=False)
        assert configured_providers() == []

    def test_unknown_provider_raises(self):
        with pytest.raises(AssetGenError):
            get_provider("nope")


# ---------------------------------------------------------------------------
# Tripo
# ---------------------------------------------------------------------------


class TestTripo:
    """Tripo V2 API payload and polling behaviour."""

    def _provider(self):
        return tripo.TripoProvider(api_key="tsk_test")

    def test_submit_text_payload(self, monkeypatch):
        captured: dict = {}

        def fake_post(url, payload, *, headers=None, timeout=60.0):
            captured["url"] = url
            captured["payload"] = payload
            return {"code": 0, "data": {"task_id": "t-123"}}

        monkeypatch.setattr(tripo, "post_json", fake_post)
        req = GenerationRequest(prompt="椅子", negative_prompt="blur", topology="quad")
        handle = self._provider().submit(req)
        assert handle.task_id == "t-123"
        assert captured["url"].endswith("/task")
        body = captured["payload"]
        assert body["type"] == "text_to_model"
        assert body["prompt"] == "椅子"
        assert body["negative_prompt"] == "blur"
        assert body["texture"] and body["pbr"]
        assert body["quad"] is True

    def test_submit_image_uses_file_token(self, monkeypatch, tmp_path):
        img = tmp_path / "ref.png"
        img.write_bytes(b"png-bytes")
        captured: dict = {}

        def fake_request(method, url, *, headers=None, body=None, timeout=60.0):
            assert method == "POST"
            assert url.endswith("/upload/sts")
            assert b"png-bytes" in body
            return json.dumps({"code": 0, "data": {"file_token": "ft-1"}}).encode()

        def fake_post(url, payload, *, headers=None, timeout=60.0):
            captured["payload"] = payload
            return {"code": 0, "data": {"task_id": "t-9"}}

        monkeypatch.setattr("cloud_robotics_sim.asset_gen.base._request", fake_request)
        monkeypatch.setattr(tripo, "post_json", fake_post)
        handle = self._provider().submit(GenerationRequest(image_path=img))
        assert handle.kind == "image"
        assert captured["payload"]["type"] == "image_to_model"
        assert captured["payload"]["file"] == {"type": "png", "file_token": "ft-1"}

    def test_poll_success_prefers_pbr(self, monkeypatch):
        def fake_get(url, *, headers=None, timeout=60.0):
            return {
                "code": 0,
                "data": {
                    "status": "success",
                    "output": {
                        "model": "https://x/base.glb",
                        "pbr_model": "https://x/pbr.glb",
                    },
                },
            }

        monkeypatch.setattr(tripo, "get_json", fake_get)
        status, url = self._provider().poll(TaskHandle("tripo", "t", "text"))
        assert status == "succeeded"
        assert url == "https://x/pbr.glb"

    def test_poll_running_and_failed(self, monkeypatch):
        responses = [
            {"code": 0, "data": {"status": "queued"}},
            {"code": 0, "data": {"status": "running"}},
            {"code": 0, "data": {"status": "failed", "error_msg": "bad prompt"}},
        ]

        def fake_get(url, *, headers=None, timeout=60.0):
            return responses.pop(0)

        monkeypatch.setattr(tripo, "get_json", fake_get)
        provider = self._provider()
        handle = TaskHandle("tripo", "t", "text")
        assert provider.poll(handle)[0] == "running"
        assert provider.poll(handle)[0] == "running"
        with pytest.raises(AssetGenError, match="bad prompt"):
            provider.poll(handle)

    def test_submit_error_envelope(self, monkeypatch):
        def fake_post(url, payload, *, headers=None, timeout=60.0):
            return {"code": 1001, "msg": "invalid key"}

        monkeypatch.setattr(tripo, "post_json", fake_post)
        with pytest.raises(AssetGenError, match="1001"):
            self._provider().submit(GenerationRequest(prompt="x"))


# ---------------------------------------------------------------------------
# Meshy
# ---------------------------------------------------------------------------


class TestMeshy:
    """Meshy API payload and polling behaviour."""

    def _provider(self):
        return meshy.MeshyProvider(api_key="msy_test")

    def test_submit_text_payload(self, monkeypatch):
        captured: dict = {}

        def fake_post(url, payload, *, headers=None, timeout=60.0):
            captured["url"] = url
            captured["payload"] = payload
            return {"result": "m-1"}

        monkeypatch.setattr(meshy, "post_json", fake_post)
        handle = self._provider().submit(
            GenerationRequest(prompt="a sword", topology="quad")
        )
        assert handle.task_id == "m-1"
        assert captured["url"].endswith("/v2/text-to-3d")
        assert captured["payload"]["mode"] == "preview"
        assert captured["payload"]["topology"] == "quad"
        assert captured["payload"]["target_formats"] == ["glb"]

    def test_submit_image_data_uri(self, monkeypatch, tmp_path):
        img = tmp_path / "ref.jpg"
        img.write_bytes(b"jpeg-bytes")
        captured: dict = {}

        def fake_post(url, payload, *, headers=None, timeout=60.0):
            captured["payload"] = payload
            return {"result": "m-2"}

        monkeypatch.setattr(meshy, "post_json", fake_post)
        self._provider().submit(GenerationRequest(image_path=img, texture=False))
        assert captured["payload"]["should_texture"] is False
        assert captured["payload"]["image_url"].startswith("data:image/jpeg;base64,")

    def test_poll_lifecycle(self, monkeypatch):
        responses = [
            {"status": "PENDING"},
            {"status": "IN_PROGRESS"},
            {"status": "SUCCEEDED", "model_urls": {"glb": "https://x/m.glb"}},
        ]

        def fake_get(url, *, headers=None, timeout=60.0):
            return responses.pop(0)

        monkeypatch.setattr(meshy, "get_json", fake_get)
        provider = self._provider()
        handle = TaskHandle("meshy", "m-2", "image")
        assert provider.poll(handle)[0] == "running"
        assert provider.poll(handle)[0] == "running"
        assert provider.poll(handle) == ("succeeded", "https://x/m.glb")

    def test_poll_failed_with_message(self, monkeypatch):
        def fake_get(url, *, headers=None, timeout=60.0):
            return {"status": "FAILED", "task_error": {"message": "busy"}}

        monkeypatch.setattr(meshy, "get_json", fake_get)
        with pytest.raises(AssetGenError, match="busy"):
            self._provider().poll(TaskHandle("meshy", "m", "text"))


# ---------------------------------------------------------------------------
# Tencent Hunyuan 3D
# ---------------------------------------------------------------------------


class TestTencentSigning:
    """TC3-HMAC-SHA256 against the official Tencent worked example."""

    """TC3-HMAC-SHA256 against the official Tencent Cloud worked example.

    Vector from cloud.tencent.com/document/api/213/30654 (signature method v3).
    """

    PAYLOAD = (
        '{"Limit": 1, "Filters": [{"Values": ["\\u672a\\u547d\\u540d"], '
        '"Name": "instance-name"}]}'
    )

    def test_hashed_payload_matches_official(self):
        assert (
            hunyuan3d.hashed_payload(self.PAYLOAD)
            == "35e9c5b0e3ae67532d3c9f17ead6c90222632e5b1ff7f6e89887f1398934f064"
        )

    def test_canonical_request_hash_matches_official(self):
        canonical = hunyuan3d.canonical_request(
            hunyuan3d.hashed_payload(self.PAYLOAD),
            "cvm.tencentcloudapi.com",
            "DescribeInstances",
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert digest == (
            "7019a55be8395899b900fb5564e4200d984910f34794a27cb3fb7d10ff6a1e84"
        )

    def test_string_to_sign_matches_official(self):
        to_sign = hunyuan3d.string_to_sign(
            1551113065,
            "2019-02-25",
            "cvm",
            "7019a55be8395899b900fb5564e4200d984910f34794a27cb3fb7d10ff6a1e84",
        )
        lines = to_sign.split("\n")
        assert lines[0] == "TC3-HMAC-SHA256"
        assert lines[1] == "1551113065"
        assert lines[2] == "2019-02-25/cvm/tc3_request"
        assert lines[3].startswith("7019a55be")

    def test_signature_deterministic_hex(self):
        to_sign = hunyuan3d.string_to_sign(1551113065, "2019-02-25", "ai3d", "ab" * 32)
        sig1 = hunyuan3d.tc3_signature("fake-secret", "2019-02-25", "ai3d", to_sign)
        sig2 = hunyuan3d.tc3_signature("fake-secret", "2019-02-25", "ai3d", to_sign)
        assert sig1 == sig2
        assert len(sig1) == 64
        int(sig1, 16)  # valid hex


class TestTencentProvider:
    """Tencent Hunyuan3D submit/poll behaviour."""

    def _provider(self):
        return hunyuan3d.TencentProvider(
            secret_id="sid", secret_key="skey", region="ap-guangzhou"
        )

    def test_submit_text_pro_job(self, monkeypatch):
        captured: dict = {}

        def fake_call(action, payload):
            captured["action"] = action
            captured["payload"] = payload
            return {"JobId": "job-1", "RequestId": "r"}

        provider = self._provider()
        monkeypatch.setattr(provider, "_call", fake_call)

        req = GenerationRequest(prompt="木椅")
        handle = provider.submit(req)
        assert handle.task_id == "job-1"
        assert captured["action"] == "SubmitHunyuanTo3DProJob"
        assert captured["payload"]["Prompt"] == "木椅"
        assert captured["payload"]["EnablePBR"] is True

    def test_submit_quad_uses_lowpoly(self, monkeypatch):
        captured: dict = {}

        def fake_call(action, payload):
            captured.update(action=action, payload=payload)
            return {"JobId": "job-2"}

        provider = self._provider()
        monkeypatch.setattr(provider, "_call", fake_call)
        provider.submit(GenerationRequest(prompt="x", topology="quad"))
        assert captured["payload"]["GenerateType"] == "LowPoly"
        assert captured["payload"]["PolygonType"] == "quadrilateral"

    def test_submit_rapid_flag_switches_actions(self, monkeypatch):
        captured: dict = {}

        def fake_call(action, payload):
            captured["action"] = action
            return {"JobId": "job-3"}

        provider = self._provider()
        monkeypatch.setattr(provider, "_call", fake_call)
        handle = provider.submit(GenerationRequest(prompt="x", extra={"rapid": True}))
        assert captured["action"] == "SubmitHunyuanTo3DRapidJob"
        assert handle.meta["rapid"] is True

    def test_submit_image_base64_strips_data_uri(self, monkeypatch, tmp_path):
        img = tmp_path / "ref.png"
        img.write_bytes(b"png")
        captured: dict = {}

        def fake_call(action, payload):
            captured["payload"] = payload
            return {"JobId": "job-4"}

        provider = self._provider()
        monkeypatch.setattr(provider, "_call", fake_call)
        provider.submit(GenerationRequest(image_path=img))
        b64 = captured["payload"]["ImageBase64"]
        assert not b64.startswith("data:")
        assert b64 == "cG5n"

    def test_poll_picks_glb(self, monkeypatch):
        responses = [
            {"Status": "WAIT"},
            {"Status": "RUN"},
            {
                "Status": "DONE",
                "ResultFile3Ds": [
                    {"Type": "OBJ", "Url": "https://x/m.obj"},
                    {"Type": "GLB", "Url": "https://x/m.glb"},
                ],
            },
        ]

        def fake_call(action, payload):
            return responses.pop(0)

        provider = self._provider()
        monkeypatch.setattr(provider, "_call", fake_call)
        handle = TaskHandle("hunyuan3d", "job-5", "text")
        assert provider.poll(handle)[0] == "running"
        assert provider.poll(handle)[0] == "running"
        assert provider.poll(handle) == ("succeeded", "https://x/m.glb")

    def test_poll_fail(self, monkeypatch):
        def fake_call(action, payload):
            return {"Status": "FAIL", "ErrorCode": "X", "ErrorMessage": "boom"}

        provider = self._provider()
        monkeypatch.setattr(provider, "_call", fake_call)
        with pytest.raises(AssetGenError, match="boom"):
            provider.poll(TaskHandle("hunyuan3d", "j", "text"))


# ---------------------------------------------------------------------------
# Rodin
# ---------------------------------------------------------------------------


class TestRodin:
    """Rodin multipart submit and status/download flow."""

    def _provider(self):
        return rodin.RodinProvider(api_key="rk")

    def test_submit_text_multipart(self, monkeypatch):
        captured: dict = {}

        def fake_request(method, url, *, headers=None, body=None, timeout=60.0):
            captured["url"] = url
            captured["body"] = body
            return json.dumps(
                {
                    "error": None,
                    "uuid": "uuid-1",
                    "jobs": {"subscription_key": "sk-1"},
                }
            ).encode()

        monkeypatch.setattr("cloud_robotics_sim.asset_gen.base._request", fake_request)
        handle = self._provider().submit(GenerationRequest(prompt="vase"))
        assert handle.task_id == "uuid-1"
        assert handle.meta["subscription_key"] == "sk-1"
        assert captured["url"].endswith("/rodin")
        body = captured["body"]
        assert b'name="prompt"' in body and b"vase" in body
        assert b"Gen-2.5-Medium" in body

    def test_poll_all_done_then_download(self, monkeypatch):
        def fake_post(url, payload, *, headers=None, timeout=60.0):
            if url.endswith("/status"):
                assert payload == {"subscription_key": "sk-1"}
                return {"error": "OK", "jobs": [{"status": "Done"}]}
            assert url.endswith("/download")
            assert payload == {"task_uuid": "uuid-1"}
            return {
                "error": None,
                "list": [
                    {"url": "https://x/preview.webp", "name": "preview.webp"},
                    {"url": "https://x/model.glb", "name": "model.glb"},
                ],
            }

        monkeypatch.setattr(rodin, "post_json", fake_post)
        handle = TaskHandle(
            "rodin", "uuid-1", "text", meta={"subscription_key": "sk-1"}
        )
        assert self._provider().poll(handle) == ("succeeded", "https://x/model.glb")

    def test_poll_failed(self, monkeypatch):
        def fake_post(url, payload, *, headers=None, timeout=60.0):
            return {"error": "OK", "jobs": [{"status": "Failed"}]}

        monkeypatch.setattr(rodin, "post_json", fake_post)
        handle = TaskHandle("rodin", "u", "text", meta={"subscription_key": "sk"})
        with pytest.raises(AssetGenError, match="failed"):
            self._provider().poll(handle)


# ---------------------------------------------------------------------------
# poll_task helper
# ---------------------------------------------------------------------------


class TestPollTask:
    """Generic polling state machine."""

    def test_succeeds(self, monkeypatch):
        monkeypatch.setattr(
            "cloud_robotics_sim.asset_gen.base.time.sleep", lambda s: None
        )
        states = iter([("running", None), ("running", None), ("succeeded", "url")])

        def poll_fn(handle):
            return next(states)

        assert (
            poll_task(TaskHandle("p", "t", "text"), poll_fn, timeout=60, interval=0)
            == "url"
        )

    def test_failed(self, monkeypatch):
        monkeypatch.setattr(
            "cloud_robotics_sim.asset_gen.base.time.sleep", lambda s: None
        )
        with pytest.raises(TaskFailedError):
            poll_task(
                TaskHandle("p", "t", "text"),
                lambda h: ("failed", None),
                timeout=60,
                interval=0,
            )

    def test_timeout(self, monkeypatch):
        monkeypatch.setattr(
            "cloud_robotics_sim.asset_gen.base.time.sleep", lambda s: None
        )
        with pytest.raises(TaskTimeoutError):
            poll_task(
                TaskHandle("p", "t", "text"),
                lambda h: ("running", None),
                timeout=1,
                interval=10,
            )


# ---------------------------------------------------------------------------
# Client end-to-end (stub provider, no network)
# ---------------------------------------------------------------------------


class _StubProvider:
    name = "stub"
    env_vars = ()

    def configured(self):
        return True

    def submit(self, request):
        return TaskHandle("stub", "task-1", "text")

    def poll(self, handle):
        return "succeeded", "https://example.com/model.glb"


class TestClient:
    """Facade end-to-end with a stub provider."""

    def test_generate_writes_glb_and_provenance(self, tmp_path, monkeypatch):
        register_provider(_StubProvider())

        def fake_download(url, dest, *, timeout=120.0):
            dest = Path(dest)
            dest.write_bytes(b"glb-bytes")
            return dest

        monkeypatch.setattr(client_module, "download_file", fake_download)
        client = AssetGenClient("stub", poll_timeout=5, poll_interval=0)
        result = client.generate_text("木椅", out_dir=tmp_path)

        assert result.model_file.read_bytes() == b"glb-bytes"
        prov = json.loads((tmp_path / "stub_task-1.json").read_text(encoding="utf-8"))
        assert prov["provider"] == "stub"
        assert prov["prompt"] == "木椅"
        assert prov["source_url"] == "https://example.com/model.glb"
        assert "created_at" in prov

    def test_auto_provider_requires_credentials(self, monkeypatch):
        for var in (
            "TRIPO_API_KEY",
            "MESHY_API_KEY",
            "RODIN_API_KEY",
            "TENCENTCLOUD_SECRET_ID",
            "TENCENTCLOUD_SECRET_KEY",
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(client_module, "_PROVIDERS", {})
        with pytest.raises(AssetGenError, match="no 3D generation provider"):
            AssetGenClient()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    """CLI entry points."""

    def test_providers_command(self, capsys, monkeypatch):
        from cloud_robotics_sim.asset_gen import __main__ as cli

        monkeypatch.delenv("TRIPO_API_KEY", raising=False)
        assert cli.main(["providers"]) == 0
        out = capsys.readouterr().out
        assert "tripo" in out and "meshy" in out
        assert "missing credentials" in out


# ---------------------------------------------------------------------------
# URL safety, size caps, and task-id sanitisation (audit H-2 / M-3)
# ---------------------------------------------------------------------------


def _stub_dns(monkeypatch, *addrs: str) -> None:
    """Route ``socket.getaddrinfo`` to fixed addresses (keeps tests offline)."""

    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [
            (
                socket.AF_INET6 if ":" in addr else socket.AF_INET,
                socket.SOCK_STREAM,
                socket.IPPROTO_TCP,
                "",
                (addr, 0, 0, 0) if ":" in addr else (addr, 0),
            )
            for addr in addrs
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


@pytest.fixture(autouse=True)
def _neutral_guard_env(monkeypatch):
    """Clear asset_gen URL-guard env overrides for every test in this module."""
    for var in (
        "CRS_ASSET_GEN_ALLOW_HTTP",
        "CRS_ASSET_GEN_ALLOW_PRIVATE",
        "CRS_ASSET_GEN_ALLOWED_HOSTS",
        "CRS_ASSET_GEN_MAX_BYTES",
    ):
        monkeypatch.delenv(var, raising=False)


class _FakeResponse:
    """Minimal urllib response stand-in: context manager + chunked ``read``."""

    def __init__(
        self,
        data: bytes,
        content_length: str | None = None,
        max_chunk: int | None = None,
    ):
        self._data = data
        self._pos = 0
        self._max_chunk = max_chunk
        self.reads: list[int] = []
        self.headers: dict[str, str] = {}
        if content_length is not None:
            self.headers["Content-Length"] = content_length

    def read(self, size: int = -1) -> bytes:
        self.reads.append(size)
        if self._max_chunk is not None and (size < 0 or size > self._max_chunk):
            size = self._max_chunk
        if size < 0:
            size = len(self._data) - self._pos
        chunk = self._data[self._pos : self._pos + size]
        self._pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class TestValidateUrl:
    """SSRF guard: scheme, userinfo, host allowlist, resolved addresses."""

    def test_rejects_file_scheme(self):
        with pytest.raises(AssetGenError, match="non-https"):
            validate_url("file:///etc/passwd")

    def test_http_rejected_without_env(self, monkeypatch):
        monkeypatch.delenv("CRS_ASSET_GEN_ALLOW_HTTP", raising=False)
        with pytest.raises(AssetGenError, match="non-https"):
            validate_url("http://10.0.0.1/redis")

    def test_http_allowed_with_env(self, monkeypatch):
        monkeypatch.setenv("CRS_ASSET_GEN_ALLOW_HTTP", "1")
        _stub_dns(monkeypatch, "93.184.216.34")
        validate_url("http://example.com/x")

    def test_rejects_userinfo(self):
        with pytest.raises(AssetGenError, match="credentials"):
            validate_url("https://user:pass@example.com/x")

    def test_rejects_missing_hostname(self):
        with pytest.raises(AssetGenError, match="hostname"):
            validate_url("https:///path/only")

    def test_rejects_literal_loopback(self):
        with pytest.raises(AssetGenError, match="non-public"):
            validate_url("https://127.0.0.1/x")

    @pytest.mark.parametrize(
        "ip",
        [
            "10.0.0.1",
            "172.16.0.9",
            "192.168.1.5",
            "169.254.169.254",
            "100.64.0.1",
            "0.0.0.0",
            "224.0.0.1",
            "::1",
            "fe80::1",
            "fc00::1",
            "ff02::1",
            "::",
        ],
    )
    def test_rejects_non_public_addresses(self, monkeypatch, ip):
        _stub_dns(monkeypatch, ip)
        with pytest.raises(AssetGenError, match="non-public"):
            validate_url("https://api.example.com/x")

    def test_rejects_when_any_resolved_address_is_private(self, monkeypatch):
        _stub_dns(monkeypatch, "93.184.216.34", "10.0.0.5")
        with pytest.raises(AssetGenError, match="non-public"):
            validate_url("https://api.example.com/x")

    def test_allows_public_addresses(self, monkeypatch):
        _stub_dns(monkeypatch, "93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946")
        validate_url("https://api.example.com/x")

    def test_private_allowed_with_env(self, monkeypatch):
        monkeypatch.setenv("CRS_ASSET_GEN_ALLOW_PRIVATE", "1")
        _stub_dns(monkeypatch, "10.0.0.5")
        validate_url("https://api.example.com/x")

    def test_allowlist(self, monkeypatch):
        monkeypatch.setenv("CRS_ASSET_GEN_ALLOWED_HOSTS", "example.com,.tripo.dev")
        _stub_dns(monkeypatch, "93.184.216.34")
        with pytest.raises(AssetGenError, match="CRS_ASSET_GEN_ALLOWED_HOSTS"):
            validate_url("https://evil.com/x")
        validate_url("https://example.com/x")
        validate_url("https://api.tripo.dev/x")

    def test_allowlist_rejects_lookalike_suffix(self, monkeypatch):
        monkeypatch.setenv("CRS_ASSET_GEN_ALLOWED_HOSTS", ".example.com")
        with pytest.raises(AssetGenError, match="CRS_ASSET_GEN_ALLOWED_HOSTS"):
            validate_url("https://notexample.com/x")


class TestResponseSizeLimits:
    """Bounded reads for API responses and streamed downloads."""

    def test_default_response_cap_is_small(self):
        assert base_module._MAX_RESPONSE_BYTES == 16 << 20

    def test_request_rejects_oversized_body(self, monkeypatch):
        _stub_dns(monkeypatch, "93.184.216.34")
        monkeypatch.setattr(base_module, "_MAX_RESPONSE_BYTES", 1024)
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda req, timeout=60.0: _FakeResponse(b"x" * 2048),
        )
        with pytest.raises(AssetGenError, match="over 1024 bytes"):
            base_module._request("GET", "https://example.com/api")

    def test_request_reads_small_body(self, monkeypatch):
        _stub_dns(monkeypatch, "93.184.216.34")
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda req, timeout=60.0: _FakeResponse(b'{"ok": 1}'),
        )
        assert base_module._request("GET", "https://example.com/api") == b'{"ok": 1}'

    def test_request_never_opens_unsafe_url(self, monkeypatch):
        opened: list[int] = []
        monkeypatch.setattr(
            "urllib.request.urlopen", lambda *args, **kwargs: opened.append(1)
        )
        with pytest.raises(AssetGenError, match="non-https"):
            base_module._request("GET", "file:///etc/passwd")
        assert opened == []

    def test_download_streams_in_chunks(self, monkeypatch, tmp_path):
        _stub_dns(monkeypatch, "93.184.216.34")
        fake_resp = _FakeResponse(b"y" * 2048, max_chunk=512)
        monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=0: fake_resp)
        dest = download_file(
            "https://example.com/m.glb", tmp_path / "m.glb", max_bytes=4096
        )
        assert dest.read_bytes() == b"y" * 2048
        assert len(fake_resp.reads) >= 4  # chunked reads, never one full read

    def test_download_truncates_over_max_bytes(self, monkeypatch, tmp_path):
        _stub_dns(monkeypatch, "93.184.216.34")
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda req, timeout=0: _FakeResponse(b"x" * 4096),
        )
        with pytest.raises(AssetGenError, match="exceeded the 1024-byte limit"):
            download_file(
                "https://example.com/m.glb", tmp_path / "m.glb", max_bytes=1024
            )
        assert not (tmp_path / "m.glb").exists()
        assert list(tmp_path.iterdir()) == []  # temp file cleaned up

    def test_download_rejects_oversized_content_length(self, monkeypatch, tmp_path):
        _stub_dns(monkeypatch, "93.184.216.34")
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda req, timeout=0: _FakeResponse(b"x" * 100, content_length="1000000"),
        )
        with pytest.raises(AssetGenError, match="declares 1000000 bytes"):
            download_file(
                "https://example.com/m.glb", tmp_path / "m.glb", max_bytes=1024
            )
        assert list(tmp_path.iterdir()) == []

    def test_download_default_max_from_env(self, monkeypatch, tmp_path):
        _stub_dns(monkeypatch, "93.184.216.34")
        monkeypatch.setenv("CRS_ASSET_GEN_MAX_BYTES", "1024")
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda req, timeout=0: _FakeResponse(b"x" * 2048),
        )
        with pytest.raises(AssetGenError, match="1024-byte limit"):
            download_file("https://example.com/m.glb", tmp_path / "m.glb")
        assert list(tmp_path.iterdir()) == []

    def test_download_rejects_file_url(self, tmp_path):
        with pytest.raises(AssetGenError, match="non-https"):
            download_file("file:///etc/passwd", tmp_path / "x.bin")


class TestSafeTaskId:
    """Filename sanitisation for provider-supplied task ids."""

    def test_clean_ids_pass_through(self):
        for raw in ("t-123", "job_1", "0", "A" * 64):
            assert safe_task_id(raw, "meshy") == raw

    def test_path_traversal_is_replaced_by_digest(self):
        digest = safe_task_id("a/../../b", "meshy")
        assert digest == hashlib.sha256(b"a/../../b").hexdigest()[:16]
        assert "/" not in digest and "\\" not in digest

    @pytest.mark.parametrize(
        "raw", ["", "a b", "x" * 65, "id\n", "../x", "a\\b", "a:b", "a/../b"]
    )
    def test_unsafe_ids_hashed(self, raw):
        out = safe_task_id(raw, "tripo")
        assert out == hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        assert all(c in "0123456789abcdef" for c in out)


class _EvilProvider:
    name = "evil"
    env_vars = ()

    def configured(self):
        return True

    def submit(self, request):
        return TaskHandle("evil", "a/../../b", "text")

    def poll(self, handle):
        return "succeeded", "https://example.com/model.glb"


class TestClientTaskIdSanitising:
    """Staging filenames cannot be escaped by a hostile task id."""

    def test_generate_contains_path_traversal(self, tmp_path, monkeypatch):
        register_provider(_EvilProvider())

        def fake_download(url, dest, *, timeout=120.0):
            dest = Path(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"glb-bytes")
            return dest

        monkeypatch.setattr(client_module, "download_file", fake_download)
        client = AssetGenClient("evil", poll_timeout=5, poll_interval=0)
        result = client.generate_text("木椅", out_dir=tmp_path)

        digest = hashlib.sha256(b"a/../../b").hexdigest()[:16]
        assert result.model_file == tmp_path / f"evil_{digest}.glb"
        assert result.model_file.read_bytes() == b"glb-bytes"
        assert result.task_id == "a/../../b"  # raw id kept for reconciliation
        assert result.metadata["sanitized_task_id"] == digest
        prov = json.loads(
            (tmp_path / f"evil_{digest}.json").read_text(encoding="utf-8")
        )
        assert prov["task_id"] == "a/../../b"
        assert not (tmp_path.parent / "b.glb").exists()
