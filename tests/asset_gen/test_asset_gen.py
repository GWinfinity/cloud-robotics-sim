"""Tests for the asset_gen 3D generation API clients (no network needed)."""

from __future__ import annotations

import hashlib
import json
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
from cloud_robotics_sim.asset_gen import client as client_module
from cloud_robotics_sim.asset_gen.base import poll_task

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
