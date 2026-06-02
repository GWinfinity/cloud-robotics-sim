"""Tests for OpenLoong Walking Controller Plugin."""

import numpy as np
import pytest

from plugins.controllers.openloong.core.walking_params import (
    WalkingParameters,
    WALKING_PRESETS,
    get_preset,
)
from plugins.controllers.openloong.core.evaluator import (
    EvaluationResult,
    WalkingEvaluator,
    evaluate_walking,
    compare_parameters,
)
from plugins.controllers.openloong.core.env import OpenLoongWalkingEnv, make_env


class TestWalkingParameters:
    """Walking parameter tests."""

    def test_init_default(self):
        """Default parameters."""
        params = WalkingParameters()
        assert params is not None

    def test_presets_exist(self):
        """Presets loaded."""
        assert len(WALKING_PRESETS) > 0

    def test_get_preset(self):
        """Preset retrieval."""
        preset = get_preset("default")
        assert preset is not None


class TestWalkingEvaluator:
    """Evaluator tests."""

    def test_init(self):
        """Evaluator initializes."""
        eval = WalkingEvaluator()
        assert eval is not None

    def test_evaluate(self):
        """Evaluation returns result."""
        params = WalkingParameters()
        result = evaluate_walking(params)
        assert isinstance(result, EvaluationResult)

    def test_compare(self):
        """Comparison returns differences."""
        p1 = WalkingParameters()
        p2 = WalkingParameters()
        diff = compare_parameters(p1, p2)
        assert diff is not None


class TestOpenLoongWalkingEnv:
    """Environment tests."""

    def test_init(self):
        """Environment initializes."""
        env = OpenLoongWalkingEnv()
        assert env is not None

    def test_make_env(self):
        """Factory function."""
        env = make_env()
        assert env is not None
