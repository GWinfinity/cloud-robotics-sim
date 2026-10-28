"""Trajectory classifier: tiny temporal models over simulated episode windows.

Pipeline position (see the trajclass discussion in AGENTS.md): Genesis rolls
out labeled episodes, :mod:`.dataset` windowizes them, :mod:`.train` fits a
~100k-param GRU that consumes dense numeric windows (something a text
triage model like laya cannot do). Genesis-free by construction — only
numpy/torch — so it runs on CPU anywhere including CI.
"""

from cloud_robotics_sim.trajclass.dataset import (
    LABEL_NAMES,
    EpisodeBatch,
    load_npz,
    save_npz,
    windows,
)
from cloud_robotics_sim.trajclass.model import TrajGRU
from cloud_robotics_sim.trajclass.train import TrainConfig, train_classifier

__all__ = [
    "LABEL_NAMES",
    "EpisodeBatch",
    "TrajGRU",
    "TrainConfig",
    "load_npz",
    "save_npz",
    "train_classifier",
    "windows",
]
