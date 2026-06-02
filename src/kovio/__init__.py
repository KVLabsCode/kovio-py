"""Kovio SDK — advertising for robots with screens."""
from .agent import KovioAgent
from .types import TaskState, SceneState, GateDecision, AdEvent
from .adapters.screen import ScreenAdapter
from .adapters.perception import PerceptionAdapter
from .cloud import CloudCampaignStore, CloudEventSink
from .config import CloudConfig, load_cloud_config

__version__ = "0.0.9"

__all__ = [
    "KovioAgent",
    "TaskState",
    "SceneState",
    "GateDecision",
    "AdEvent",
    "ScreenAdapter",
    "PerceptionAdapter",
    "CloudCampaignStore",
    "CloudEventSink",
    "CloudConfig",
    "load_cloud_config",
    "__version__",
]
