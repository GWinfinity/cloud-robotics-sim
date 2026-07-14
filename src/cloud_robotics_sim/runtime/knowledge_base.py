"""Knowledge base for recording successful and failed improvement patterns."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .experiments import ValidationResult
from .proposals import ImprovementProposal

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeEntry:
    """One record of an attempted improvement."""

    diagnosis_category: str
    proposal_name: str
    passed: bool
    confidence: float
    improvement: dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=__import__("time").time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagnosis_category": self.diagnosis_category,
            "proposal_name": self.proposal_name,
            "passed": self.passed,
            "confidence": self.confidence,
            "improvement": self.improvement,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeEntry":
        return cls(**data)


class KnowledgeBase:
    """Store and query historical improvement attempts."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else Path("knowledge_base.json")
        self.entries: list[KnowledgeEntry] = []
        self._load()

    def record(self, result: ValidationResult) -> None:
        """Record a validation result."""
        entry = KnowledgeEntry(
            diagnosis_category=result.proposal.target_diagnosis,
            proposal_name=result.proposal.name,
            passed=result.passed,
            confidence=result.confidence,
            improvement=result.improvement,
        )
        self.entries.append(entry)
        self._save()

    def has_failed(self, proposal: ImprovementProposal, min_attempts: int = 2) -> bool:
        """Return True if this proposal has failed at least min_attempts times."""
        failures = [
            e for e in self.entries if e.proposal_name == proposal.name and not e.passed
        ]
        return len(failures) >= min_attempts

    def best_for(self, diagnosis_category: str) -> KnowledgeEntry | None:
        """Return the best passing entry for a diagnosis category."""
        passing = [
            e
            for e in self.entries
            if e.diagnosis_category == diagnosis_category and e.passed
        ]
        if not passing:
            return None
        return max(passing, key=lambda e: e.confidence)

    def summary(self) -> dict[str, Any]:
        """Return a summary of the knowledge base."""
        total = len(self.entries)
        passed = sum(1 for e in self.entries if e.passed)
        return {
            "total_attempts": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total else 0.0,
        }

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump([e.to_dict() for e in self.entries], f, indent=2)

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            self.entries = [KnowledgeEntry.from_dict(d) for d in data]
        except Exception as e:
            logger.warning(f"Failed to load knowledge base: {e}")
