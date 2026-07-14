"""Tests for device selection utilities."""

from types import SimpleNamespace

from cloud_robotics_sim.utils import device as device_utils


class TestAvailability:
    """Tests for device availability flags."""

    def test_has_torch(self):
        """HAS_TORCH should reflect whether torch is installed."""
        assert isinstance(device_utils.HAS_TORCH, bool)

    def test_musa_unavailable_without_torch_musa(self, monkeypatch):
        """is_musa_available should be False when torch_musa is not present."""
        monkeypatch.setattr(device_utils, "HAS_MUSA", False)
        assert device_utils.is_musa_available() is False


class TestDefaultDevice:
    """Tests for default device selection."""

    def test_default_device_env_cpu(self, monkeypatch):
        """CRS_DEVICE=cpu should force CPU."""
        monkeypatch.setenv("CRS_DEVICE", "cpu")
        assert device_utils.default_device() == "cpu"

    def test_default_device_env_musa_not_available(self, monkeypatch):
        """CRS_DEVICE=musa without musa should fall through to cuda/cpu."""
        monkeypatch.setenv("CRS_DEVICE", "musa")
        monkeypatch.setattr(device_utils, "is_musa_available", lambda: False)
        monkeypatch.setattr(device_utils, "is_cuda_available", lambda: False)
        assert device_utils.default_device() == "cpu"

    def test_default_device_prefers_musa(self, monkeypatch):
        """If MUSA is available, default_device should return musa."""
        monkeypatch.delenv("CRS_DEVICE", raising=False)
        monkeypatch.setattr(device_utils, "is_musa_available", lambda: True)
        monkeypatch.setattr(device_utils, "is_cuda_available", lambda: True)
        assert device_utils.default_device() == "musa"

    def test_default_device_prefers_cuda(self, monkeypatch):
        """If only CUDA is available, default_device should return cuda."""
        monkeypatch.delenv("CRS_DEVICE", raising=False)
        monkeypatch.setattr(device_utils, "is_musa_available", lambda: False)
        monkeypatch.setattr(device_utils, "is_cuda_available", lambda: True)
        assert device_utils.default_device() == "cuda"

    def test_default_device_fallback_cpu(self, monkeypatch):
        """If no accelerator is available, default_device should return cpu."""
        monkeypatch.delenv("CRS_DEVICE", raising=False)
        monkeypatch.setattr(device_utils, "is_musa_available", lambda: False)
        monkeypatch.setattr(device_utils, "is_cuda_available", lambda: False)
        assert device_utils.default_device() == "cpu"


class TestGetDevice:
    """Tests for explicit device preference resolution."""

    def test_get_device_musa_available(self, monkeypatch):
        """get_device('musa') should return musa when available."""
        monkeypatch.setattr(device_utils, "is_musa_available", lambda: True)
        assert device_utils.get_device("musa") == "musa"

    def test_get_device_musa_unavailable_fallback(self, monkeypatch):
        """get_device('musa') should fall back when MUSA is unavailable."""
        monkeypatch.setattr(device_utils, "is_musa_available", lambda: False)
        monkeypatch.setattr(device_utils, "is_cuda_available", lambda: False)
        assert device_utils.get_device("musa") == "cpu"

    def test_get_device_cpu(self):
        """get_device('cpu') should always return cpu."""
        assert device_utils.get_device("cpu") == "cpu"


class TestSetDefaultDevice:
    """Tests for setting the default torch device."""

    def test_set_default_device_cpu(self):
        """CPU device should be returned without touching torch defaults."""
        assert device_utils.set_default_device("cpu") == "cpu"

    def test_set_default_device_musa_no_crash(self, monkeypatch):
        """set_default_device should not crash when MUSA is unavailable."""
        monkeypatch.setattr(device_utils, "is_musa_available", lambda: False)
        assert device_utils.set_default_device("musa") == "cpu"


class TestDeviceFlags:
    """Tests for module-level availability flags."""

    def test_has_torch_is_bool(self):
        """HAS_TORCH should be a boolean."""
        assert isinstance(device_utils.HAS_TORCH, bool)

    def test_has_cuda_is_bool(self):
        """HAS_CUDA should be a boolean."""
        assert isinstance(device_utils.HAS_CUDA, bool)

    def test_has_musa_is_bool(self):
        """HAS_MUSA should be a boolean."""
        assert isinstance(device_utils.HAS_MUSA, bool)


class TestDeviceExceptionBranches:
    """Tests for device helper exception branches."""

    def test_is_musa_available_exception(self, monkeypatch):
        """is_musa_available should return False when torch.musa raises."""
        monkeypatch.setattr(device_utils, "HAS_MUSA", True)
        monkeypatch.setattr(device_utils, "HAS_TORCH", True)

        fake_torch = SimpleNamespace(
            musa=SimpleNamespace(
                is_available=lambda: (_ for _ in ()).throw(RuntimeError("fail"))
            )
        )
        monkeypatch.setattr(device_utils, "torch", fake_torch)
        assert device_utils.is_musa_available() is False

    def test_is_cuda_available_exception(self, monkeypatch):
        """is_cuda_available should return False when torch.cuda raises."""
        monkeypatch.setattr(device_utils, "HAS_CUDA", True)
        monkeypatch.setattr(device_utils, "HAS_TORCH", True)

        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: (_ for _ in ()).throw(RuntimeError("fail"))
            )
        )
        monkeypatch.setattr(device_utils, "torch", fake_torch)
        assert device_utils.is_cuda_available() is False

    def test_set_default_device_exception(self, monkeypatch):
        """set_default_device should warn and return device when torch.set_default_device fails."""
        monkeypatch.setattr(device_utils, "HAS_TORCH", True)

        class FakeTorch:
            @staticmethod
            def set_default_device(_device):
                raise RuntimeError("fail")

        monkeypatch.setattr(device_utils, "torch", FakeTorch())
        monkeypatch.setattr(device_utils, "get_device", lambda _pref: "cuda")
        assert device_utils.set_default_device("cuda") == "cuda"
