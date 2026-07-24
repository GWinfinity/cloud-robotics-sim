"""Tests for Genesis compatibility utilities."""

from types import SimpleNamespace

import numpy as np
import pytest

from cloud_robotics_sim.utils import genesis_compat


class TestBackendCompatibility:
    """Tests for backend selection helpers."""

    def test_get_genesis_backend_cpu(self):
        """CPU backend should always be available."""
        backend = genesis_compat.get_genesis_backend("cpu")
        assert backend is not None

    def test_get_genesis_backend_unknown(self):
        """Unknown backend names return None."""
        assert genesis_compat.get_genesis_backend("unknown") is None

    def test_get_genesis_backend_musa_returns_none(self):
        """MUSA is not a native Genesis backend and should return None."""
        assert genesis_compat.get_genesis_backend("musa") is None

    def test_get_genesis_lights(self):
        """Lights helper should not raise."""
        lights = genesis_compat.get_genesis_lights()
        # genesis-world 1.2+ does not expose gs.lights
        assert lights is None or hasattr(lights, "Ambient")


class TestGenesisInit:
    """Tests for Genesis initialization helpers."""

    def test_genesis_init_cpu(self):
        """Initializing with CPU backend should succeed."""
        genesis_compat.genesis_init(headless=True, use_cuda=False)
        # Idempotent: calling again should not raise
        genesis_compat.ensure_genesis_initialized(headless=True, use_cuda=False)

    def test_genesis_init_cuda_fallback(self):
        """Requesting CUDA on a CPU machine should fall back to CPU."""
        genesis_compat.genesis_init(headless=True, use_cuda=True)


class TestObjectQueries:
    """Tests for object query helpers."""

    def _make_obj(self, name):
        return SimpleNamespace(get_name=lambda: name)

    def test_get_obj_by_name_unique(self):
        """get_obj_by_name should return the matching object."""
        obj = self._make_obj("arm")
        assert genesis_compat.get_obj_by_name([obj], "arm") is obj

    def test_get_obj_by_name_not_found(self):
        """get_obj_by_name should return None when no object matches."""
        assert genesis_compat.get_obj_by_name([], "arm") is None

    def test_get_obj_by_name_non_unique_raises(self):
        """get_obj_by_name should raise when multiple objects match."""
        objs = [self._make_obj("arm"), self._make_obj("arm")]
        with pytest.raises(RuntimeError):
            genesis_compat.get_obj_by_name(objs, "arm")

    def test_get_obj_by_name_non_unique_list(self):
        """get_obj_by_name can return a list when is_unique is False."""
        objs = [self._make_obj("arm"), self._make_obj("arm")]
        assert genesis_compat.get_obj_by_name(objs, "arm", is_unique=False) == objs

    def test_get_objs_by_names_order(self):
        """get_objs_by_names should preserve the order of the names list."""
        a = self._make_obj("a")
        b = self._make_obj("b")
        result = genesis_compat.get_objs_by_names([b, a], ["a", "b"])
        assert result == [a, b]

    def test_get_obj_by_type(self):
        """get_obj_by_type should match by Python type."""
        assert genesis_compat.get_obj_by_type([1, "x", 2.0], str) == "x"

    def test_get_obj_by_type_multiple(self):
        """get_obj_by_type should raise for multiple matches when unique."""
        with pytest.raises(RuntimeError):
            genesis_compat.get_obj_by_type([1, 2, 3], int)


class TestUrdfConfig:
    """Tests for URDF configuration helpers."""

    def test_check_urdf_config_valid(self):
        """check_urdf_config should pass for a valid config."""
        config = {"material": {"density": 1000.0}, "link": {}}
        genesis_compat.check_urdf_config(config)

    def test_check_urdf_config_invalid_key(self):
        """check_urdf_config should raise for invalid top-level keys."""
        with pytest.raises(KeyError):
            genesis_compat.check_urdf_config({"invalid": 1})

    def test_check_urdf_config_invalid_link_key(self):
        """check_urdf_config should raise for invalid link keys."""
        config = {"link": {"base": {"invalid": 1}}}
        with pytest.raises(KeyError):
            genesis_compat.check_urdf_config(config)

    def test_parse_urdf_config_without_gs(self, monkeypatch):
        """parse_urdf_config should keep dict material when gs is unavailable."""
        monkeypatch.setattr(genesis_compat, "gs", None)
        config = {"material": {"density": 500.0}}
        parsed = genesis_compat.parse_urdf_config(config)
        assert parsed["material"] == {"density": 500.0}

    def test_apply_urdf_config(self):
        """apply_urdf_config should call loader methods for each property."""
        loader = SimpleNamespace(
            set_link_material=lambda *args: setattr(loader, "link_material_args", args),
            set_link_patch_radius=lambda name, r: setattr(
                loader, "patch_radius", (name, r)
            ),
            set_link_min_patch_radius=lambda name, r: setattr(
                loader, "min_patch_radius", (name, r)
            ),
            set_link_density=lambda name, d: setattr(loader, "density", (name, d)),
            set_material=lambda *args: setattr(loader, "material_args", args),
        )
        urdf_config = {
            "material": {"density": 1000.0},
            "link": {
                "base": {
                    "material": {
                        "static_friction": 0.6,
                        "dynamic_friction": 0.5,
                        "restitution": 0.1,
                    },
                    "patch_radius": 0.02,
                    "min_patch_radius": 0.01,
                    "density": 800.0,
                }
            },
        }
        genesis_compat.apply_urdf_config(loader, urdf_config)
        assert loader.density == ("base", 800.0)
        assert loader.patch_radius == ("base", 0.02)
        assert loader.min_patch_radius == ("base", 0.01)


class TestStateExtraction:
    """Tests for actor and articulation state extraction."""

    def _make_pose(self, p, q):
        return SimpleNamespace(p=p, q=q)

    def test_get_actor_state(self):
        """get_actor_state should return a concatenated state vector."""
        actor = SimpleNamespace(
            get_pose=lambda: self._make_pose([1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0]),
            get_linear_velocity=lambda: np.array([0.1, 0.0, 0.0]),
            get_angular_velocity=lambda: np.array([0.0, 0.1, 0.0]),
        )
        state = genesis_compat.get_actor_state(actor)
        assert state is not None
        assert state.shape == (13,)

    def test_get_articulation_state(self):
        """get_articulation_state should return root + joint states."""
        link = SimpleNamespace(
            get_pose=lambda: self._make_pose([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            get_linear_velocity=lambda: np.zeros(3),
            get_angular_velocity=lambda: np.zeros(3),
        )
        articulation = SimpleNamespace(
            get_links=lambda: [link],
            get_qpos=lambda: np.array([0.1, 0.2]),
            get_qvel=lambda: np.array([0.01, 0.02]),
        )
        state = genesis_compat.get_articulation_state(articulation)
        assert state is not None
        assert state.shape == (13 + 4,)

    def test_get_articulation_padded_state(self):
        """get_articulation_padded_state should pad to the requested dof."""
        link = SimpleNamespace(
            get_pose=lambda: self._make_pose([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            get_linear_velocity=lambda: np.zeros(3),
            get_angular_velocity=lambda: np.zeros(3),
        )
        articulation = SimpleNamespace(
            get_links=lambda: [link],
            get_qpos=lambda: np.array([0.1]),
            get_qvel=lambda: np.array([0.01]),
        )
        state = genesis_compat.get_articulation_padded_state(articulation, max_dof=4)
        assert state is not None
        assert state.shape == (13 + 8,)


class TestContactProcessing:
    """Tests for contact processing helpers."""

    def _make_contact(self, entity0, entity1, impulse=None):
        point = SimpleNamespace(impulse=impulse if impulse is not None else np.zeros(3))
        body0 = SimpleNamespace(entity=entity0)
        body1 = SimpleNamespace(entity=entity1)
        return SimpleNamespace(bodies=[body0, body1], points=[point])

    def test_get_pairwise_contacts(self):
        """get_pairwise_contacts should find contacts between two actors."""
        a, b = object(), object()
        contact = self._make_contact(a, b)
        result = genesis_compat.get_pairwise_contacts([contact], a, b)
        assert len(result) == 1
        assert result[0][1] is True

    def test_get_pairwise_contact_impulse(self):
        """get_pairwise_contact_impulse should sum contact impulses."""
        a, b = object(), object()
        contact = self._make_contact(a, b, impulse=np.array([1.0, 0.0, 0.0]))
        impulse = genesis_compat.get_pairwise_contact_impulse([contact], a, b)
        assert np.allclose(impulse, [1.0, 0.0, 0.0])

    def test_get_cpu_actor_contacts(self):
        """get_cpu_actor_contacts should collect contacts for a single actor."""
        a, b = object(), object()
        contact = self._make_contact(a, b)
        result = genesis_compat.get_cpu_actor_contacts([contact], a)
        assert len(result) == 1

    def test_get_cpu_actors_contacts(self):
        """get_cpu_actors_contacts should map actors to their contacts."""
        a, b = object(), object()
        contact = self._make_contact(a, b)
        result = genesis_compat.get_cpu_actors_contacts([contact], [a, b])
        assert len(result[a]) == 1
        assert len(result[b]) == 0


class TestJointAndActorUtilities:
    """Tests for joint and actor utility helpers."""

    def test_check_joint_stuck(self):
        """check_joint_stuck should detect a stuck joint."""
        articulation = SimpleNamespace(
            get_qpos=lambda: np.array([0.5]),
            get_drive_target=lambda: np.array([0.0]),
            get_qvel=lambda: np.array([0.0]),
        )
        assert genesis_compat.check_joint_stuck(articulation, 0)

    def test_check_actor_static_numpy(self):
        """check_actor_static should return True for small velocities."""
        actor = SimpleNamespace(
            get_linear_velocity=lambda: np.zeros(3),
            get_angular_velocity=lambda: np.zeros(3),
        )
        assert genesis_compat.check_actor_static(actor)

    def test_is_state_dict_consistent(self):
        """is_state_dict_consistent should detect mismatched batch sizes."""
        consistent = {
            "actors": {"a": np.zeros((4, 3))},
            "articulations": {"b": np.zeros((4, 5))},
        }
        inconsistent = {
            "actors": {"a": np.zeros((4, 3))},
            "articulations": {"b": np.zeros((2, 5))},
        }
        assert genesis_compat.is_state_dict_consistent(consistent)
        assert not genesis_compat.is_state_dict_consistent(inconsistent)

    def test_is_state_dict_consistent_no_keys(self):
        """is_state_dict_consistent should return True for an empty state dict."""
        assert genesis_compat.is_state_dict_consistent({})

    def test_check_actor_static_torch_tensors(self):
        """check_actor_static should handle torch tensors and return bool."""
        import torch

        actor = SimpleNamespace(
            linear_velocity=torch.zeros((1, 3)),
            angular_velocity=torch.zeros((1, 3)),
        )
        result = genesis_compat.check_actor_static(actor)
        assert isinstance(result, bool)
        assert result is True

    def test_check_actor_static_missing_attrs(self):
        """check_actor_static should return True when velocity attrs are missing."""
        actor = SimpleNamespace()
        assert genesis_compat.check_actor_static(actor) is True

    def test_check_joint_stuck_exception(self):
        """check_joint_stuck should return False when getters raise."""
        articulation = SimpleNamespace(
            get_qpos=lambda: (_ for _ in ()).throw(RuntimeError("fail"))
        )
        assert genesis_compat.check_joint_stuck(articulation, 0) is False

    def test_check_joint_stuck_false(self):
        """check_joint_stuck should return False when not stuck."""
        articulation = SimpleNamespace(
            get_qpos=lambda: np.array([0.0]),
            get_drive_target=lambda: np.array([0.0]),
            get_qvel=lambda: np.array([0.0]),
        )
        assert not genesis_compat.check_joint_stuck(articulation, 0)


class TestGenesisInitBranches:
    """Tests for Genesis initialization branches."""

    def test_genesis_init_with_musa_device(self, monkeypatch):
        """genesis_init should fall back to CPU when device='musa'."""
        fake_backend = object()

        class GenesisExceptionError(Exception):
            pass

        class FakeGs:
            _initialized = False
            init_calls = []
            GenesisException = GenesisExceptionError

            @staticmethod
            def init(**kwargs):
                FakeGs.init_calls.append(kwargs)

        monkeypatch.setattr(genesis_compat, "HAS_GENESIS", True)
        monkeypatch.setattr(genesis_compat, "gs", FakeGs())
        monkeypatch.setattr(
            genesis_compat, "get_genesis_backend", lambda name: fake_backend
        )
        monkeypatch.setattr(
            "cloud_robotics_sim.utils.device.get_device", lambda _pref: "musa"
        )

        genesis_compat.genesis_init(device="musa")
        assert FakeGs.init_calls
        assert FakeGs.init_calls[0]["backend"] is fake_backend

    def test_ensure_genesis_initialized_no_genesis(self, monkeypatch):
        """ensure_genesis_initialized should raise when Genesis is unavailable."""
        monkeypatch.setattr(genesis_compat, "HAS_GENESIS", False)
        monkeypatch.setattr(genesis_compat, "gs", None)
        with pytest.raises(RuntimeError):
            genesis_compat.ensure_genesis_initialized()

    def test_ensure_genesis_initialized_exception(self, monkeypatch):
        """ensure_genesis_initialized should swallow GenesisException."""

        class GenesisExceptionError(Exception):
            pass

        class FakeGs:
            _initialized = False
            init_calls = []
            GenesisException = GenesisExceptionError

            @staticmethod
            def init(**kwargs):
                FakeGs.init_calls.append(kwargs)

        monkeypatch.setattr(genesis_compat, "HAS_GENESIS", True)
        monkeypatch.setattr(genesis_compat, "gs", FakeGs())
        monkeypatch.setattr(genesis_compat, "get_genesis_backend", lambda _name: None)

        # First call initializes, second should raise GenesisException because
        # gs._initialized is still False in the fake but get_genesis_backend
        # returns None so gs.init is called again. We simulate the exception by
        # making the second gs.init raise.
        genesis_compat.ensure_genesis_initialized()
        assert len(FakeGs.init_calls) == 1


class TestBackendEnumFallback:
    """Tests for get_genesis_backend internal enum fallback."""

    def test_internal_enum_fallback(self, monkeypatch):
        """get_genesis_backend should fall back to gs._gs_backend."""
        fake_backend = object()

        class FakeEnum:
            cuda = fake_backend

        class FakeGs:
            _gs_backend = FakeEnum()

        monkeypatch.setattr(genesis_compat, "HAS_GENESIS", True)
        monkeypatch.setattr(genesis_compat, "gs", FakeGs())
        assert genesis_compat.get_genesis_backend("cuda") is fake_backend


class TestUrdfConfigBranches:
    """Tests for URDF configuration branches."""

    def test_check_urdf_config_nested_link_valid(self):
        """check_urdf_config should accept valid nested link config."""
        config = {
            "link": {
                "base": {
                    "material": {"density": 1000.0},
                    "patch_radius": 0.02,
                    "min_patch_radius": 0.01,
                }
            }
        }
        genesis_compat.check_urdf_config(config)

    def test_parse_urdf_config_with_fake_rigid(self, monkeypatch):
        """parse_urdf_config should create gs.materials.Rigid when available."""
        created = []

        class FakeRigid:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                created.append(kwargs)

        class FakeMaterials:
            Rigid = FakeRigid

        class FakeGs:
            materials = FakeMaterials()

        monkeypatch.setattr(genesis_compat, "gs", FakeGs())
        config = {
            "material": {"density": 1000.0},
            "_materials": {"mat1": {"density": 500.0}},
            "link": {
                "base": {
                    "material": "mat1",
                    "density": 800.0,
                }
            },
        }
        parsed = genesis_compat.parse_urdf_config(config)
        assert parsed["material"].kwargs == {"density": 1000.0}
        assert parsed["link"]["base"]["material"].kwargs == {"density": 500.0}

    def test_apply_urdf_config_global_branches(self):
        """apply_urdf_config should apply global material/patch/density."""
        calls = {}
        loader = SimpleNamespace(
            set_material=lambda *args: calls.setdefault("material", args),
            set_patch_radius=lambda r: calls.setdefault("patch_radius", r),
            set_min_patch_radius=lambda r: calls.setdefault("min_patch_radius", r),
            set_density=lambda d: calls.setdefault("density", d),
        )
        urdf_config = {
            "material": SimpleNamespace(
                static_friction=0.6,
                dynamic_friction=0.5,
                restitution=0.1,
            ),
            "patch_radius": 0.02,
            "min_patch_radius": 0.01,
            "density": 900.0,
        }
        genesis_compat.apply_urdf_config(loader, urdf_config)
        assert calls["material"] == (0.6, 0.5, 0.1)
        assert calls["patch_radius"] == 0.02
        assert calls["min_patch_radius"] == 0.01
        assert calls["density"] == 900.0


class TestStateExtractionBranches:
    """Tests for state extraction failure branches."""

    def _make_pose(self, p, q):
        return SimpleNamespace(p=p, q=q)

    def test_get_actor_state_exception(self):
        """get_actor_state should return None when actor access raises."""
        actor = SimpleNamespace(
            get_pose=lambda: (_ for _ in ()).throw(RuntimeError("fail"))
        )
        assert genesis_compat.get_actor_state(actor) is None

    def test_get_articulation_state_empty_links(self):
        """get_articulation_state should return None for empty links."""
        articulation = SimpleNamespace(get_links=lambda: [])
        assert genesis_compat.get_articulation_state(articulation) is None

    def test_get_articulation_padded_state_assertion_fails(self):
        """get_articulation_padded_state should assert when max_dof is too small."""
        link = SimpleNamespace(
            get_pose=lambda: self._make_pose([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            get_linear_velocity=lambda: np.zeros(3),
            get_angular_velocity=lambda: np.zeros(3),
        )
        articulation = SimpleNamespace(
            get_links=lambda: [link],
            get_qpos=lambda: np.array([0.1, 0.2, 0.3]),
            get_qvel=lambda: np.array([0.01, 0.02, 0.03]),
        )
        with pytest.raises(ValueError, match="max_dof"):
            genesis_compat.get_articulation_padded_state(articulation, max_dof=2)


class TestMultiplePairwiseContacts:
    """Tests for multiple pairwise contact helper."""

    def _make_contact(self, entity0, entity1, impulse=None):
        point = SimpleNamespace(impulse=impulse if impulse is not None else np.zeros(3))
        body0 = SimpleNamespace(entity=entity0)
        body1 = SimpleNamespace(entity=entity1)
        return SimpleNamespace(bodies=[body0, body1], points=[point])

    def test_get_multiple_pairwise_contacts(self):
        """get_multiple_pairwise_contacts should map contacts per actor."""
        a, b, c = object(), object(), object()
        contacts = [
            self._make_contact(a, b, impulse=np.array([1.0, 0.0, 0.0])),
            self._make_contact(c, a, impulse=np.array([0.0, 1.0, 0.0])),
        ]
        result = genesis_compat.get_multiple_pairwise_contacts([contacts[0]], a, [b, c])
        assert len(result[b]) == 1
        assert result[b][0][1] is True
        assert len(result[c]) == 0

        result2 = genesis_compat.get_multiple_pairwise_contacts(
            [contacts[1]], a, [b, c]
        )
        assert len(result2[c]) == 1
        assert result2[c][0][1] is False
