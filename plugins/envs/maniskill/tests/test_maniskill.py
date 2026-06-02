"""Tests for ManiSkill Environment Plugin."""

import numpy as np
import pytest

from plugins.envs.maniskill.core.genesis_maniskill.tasks import list_available_tasks, get_task, TASK_REGISTRY
from plugins.envs.maniskill.core.genesis_maniskill.agents import list_available_agents, get_agent


class TestTaskRegistry:
    """Task registry tests."""

    def test_list_tasks(self):
        """Task listing works."""
        tasks = list_available_tasks()
        assert isinstance(tasks, dict)
        assert len(tasks) > 0

    def test_task_registry(self):
        """Task registry has entries."""
        assert 'pick_place' in TASK_REGISTRY
        assert 'push' in TASK_REGISTRY


class TestAgentRegistry:
    """Agent registry tests."""

    def test_list_agents(self):
        """Agent listing works."""
        agents = list_available_agents()
        assert isinstance(agents, dict)
        assert len(agents) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
