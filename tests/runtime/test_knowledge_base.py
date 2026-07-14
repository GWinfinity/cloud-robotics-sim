"""Tests for runtime knowledge base."""

import pytest

from cloud_robotics_sim.runtime.experiments import ValidationResult
from cloud_robotics_sim.runtime.knowledge_base import KnowledgeBase, KnowledgeEntry
from cloud_robotics_sim.runtime.metrics import MetricSummary
from cloud_robotics_sim.runtime.proposals import ImprovementProposal


class TestKnowledgeEntry:
    """Tests for KnowledgeEntry."""

    def test_to_dict(self):
        entry = KnowledgeEntry(
            diagnosis_category="x",
            proposal_name="p",
            passed=True,
            confidence=0.9,
            improvement={"reward": 5.0},
            timestamp=123.0,
        )
        d = entry.to_dict()
        assert d["diagnosis_category"] == "x"
        assert d["proposal_name"] == "p"
        assert d["passed"] is True
        assert d["confidence"] == pytest.approx(0.9)
        assert d["improvement"] == {"reward": pytest.approx(5.0)}
        assert d["timestamp"] == pytest.approx(123.0)

    def test_from_dict(self):
        data = {
            "diagnosis_category": "x",
            "proposal_name": "p",
            "passed": False,
            "confidence": 0.2,
            "improvement": {},
            "timestamp": 456.0,
        }
        entry = KnowledgeEntry.from_dict(data)
        assert entry.diagnosis_category == "x"
        assert entry.proposal_name == "p"
        assert entry.passed is False
        assert entry.confidence == pytest.approx(0.2)
        assert entry.timestamp == pytest.approx(456.0)


class TestKnowledgeBaseRecord:
    """Tests for KnowledgeBase.record."""

    def test_record(self, tmp_path):
        kb = KnowledgeBase(path=tmp_path / "kb.json")
        proposal = ImprovementProposal(
            name="p", description="", target_diagnosis="cat", config_delta={}
        )
        result = ValidationResult(
            proposal=proposal,
            baseline_summary=MetricSummary(),
            proposal_summary=MetricSummary(),
            improvement={"reward": 5.0},
            passed=True,
            confidence=0.9,
            reason="ok",
        )
        kb.record(result)
        assert len(kb.entries) == 1
        assert kb.entries[0].proposal_name == "p"
        assert kb.entries[0].passed is True
        assert (tmp_path / "kb.json").exists()


class TestKnowledgeBaseQueries:
    """Tests for KnowledgeBase query methods."""

    @pytest.fixture
    def kb(self, tmp_path):
        return KnowledgeBase(path=tmp_path / "kb.json")

    def test_has_failed_false_when_not_enough_attempts(self, kb):
        proposal = ImprovementProposal(
            name="p", description="", target_diagnosis="cat", config_delta={}
        )
        result = ValidationResult(
            proposal=proposal,
            baseline_summary=MetricSummary(),
            proposal_summary=MetricSummary(),
            passed=False,
            confidence=0.0,
            reason="bad",
        )
        kb.record(result)
        assert kb.has_failed(proposal, min_attempts=2) is False

    def test_has_failed_true_when_enough_attempts(self, kb):
        proposal = ImprovementProposal(
            name="p", description="", target_diagnosis="cat", config_delta={}
        )
        result = ValidationResult(
            proposal=proposal,
            baseline_summary=MetricSummary(),
            proposal_summary=MetricSummary(),
            passed=False,
            confidence=0.0,
            reason="bad",
        )
        kb.record(result)
        kb.record(result)
        assert kb.has_failed(proposal, min_attempts=2) is True

    def test_best_for_returns_none_when_no_passing(self, kb):
        proposal = ImprovementProposal(
            name="p", description="", target_diagnosis="cat", config_delta={}
        )
        result = ValidationResult(
            proposal=proposal,
            baseline_summary=MetricSummary(),
            proposal_summary=MetricSummary(),
            passed=False,
            confidence=0.0,
            reason="bad",
        )
        kb.record(result)
        assert kb.best_for("cat") is None

    def test_best_for_returns_highest_confidence(self, kb):
        for conf, passed in [(0.5, True), (0.9, True), (0.2, False)]:
            proposal = ImprovementProposal(
                name=f"p{conf}", description="", target_diagnosis="cat", config_delta={}
            )
            result = ValidationResult(
                proposal=proposal,
                baseline_summary=MetricSummary(),
                proposal_summary=MetricSummary(),
                passed=passed,
                confidence=conf,
                reason="ok",
            )
            kb.record(result)
        best = kb.best_for("cat")
        assert best is not None
        assert best.confidence == pytest.approx(0.9)

    def test_summary(self, kb):
        proposal = ImprovementProposal(
            name="p", description="", target_diagnosis="cat", config_delta={}
        )
        for passed in [True, False, True]:
            result = ValidationResult(
                proposal=proposal,
                baseline_summary=MetricSummary(),
                proposal_summary=MetricSummary(),
                passed=passed,
                confidence=0.5,
                reason="ok",
            )
            kb.record(result)
        summary = kb.summary()
        assert summary["total_attempts"] == 3
        assert summary["passed"] == 2
        assert summary["failed"] == 1
        assert summary["pass_rate"] == pytest.approx(2 / 3)

    def test_summary_empty(self, kb):
        summary = kb.summary()
        assert summary["total_attempts"] == 0
        assert summary["pass_rate"] == pytest.approx(0.0)


class TestKnowledgeBaseLoad:
    """Tests for KnowledgeBase._load edge cases."""

    def test_load_empty_file(self, tmp_path):
        path = tmp_path / "kb.json"
        path.write_text("")
        kb = KnowledgeBase(path=path)
        assert kb.entries == []

    def test_load_corrupted_file(self, tmp_path):
        path = tmp_path / "kb.json"
        path.write_text("not valid json")
        kb = KnowledgeBase(path=path)
        assert kb.entries == []

    def test_load_existing_entries(self, tmp_path):
        path = tmp_path / "kb.json"
        path.write_text(
            '[{"diagnosis_category": "x", "proposal_name": "p", "passed": true, '
            '"confidence": 0.8, "improvement": {}, "timestamp": 1.0}]'
        )
        kb = KnowledgeBase(path=path)
        assert len(kb.entries) == 1
        assert kb.entries[0].proposal_name == "p"


class TestKnowledgeBaseSave:
    """Tests for KnowledgeBase._save."""

    def test_save_creates_directories(self, tmp_path):
        nested = tmp_path / "a" / "b" / "kb.json"
        kb = KnowledgeBase(path=nested)
        proposal = ImprovementProposal(
            name="p", description="", target_diagnosis="cat", config_delta={}
        )
        result = ValidationResult(
            proposal=proposal,
            baseline_summary=MetricSummary(),
            proposal_summary=MetricSummary(),
            passed=True,
            confidence=0.8,
            reason="ok",
        )
        kb.record(result)
        assert nested.exists()
