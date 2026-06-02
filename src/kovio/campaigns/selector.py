"""CampaignSelector — choose what plays right now.

Abstract base + a `RuleBasedSelector` default that uses AND-of-predicates
matching, priority ordering, and per-campaign encounter caps. Plug in your
own selector if you ever need ML scoring or multi-armed bandit logic.
"""
from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod

from .models import Campaign, DecisionContext
from .store import CampaignStore

log = logging.getLogger("kovio.selector")


class CampaignSelector(ABC):
    @abstractmethod
    def select(self, ctx: DecisionContext) -> Campaign | None:
        ...

    @abstractmethod
    def record_play(self, campaign_id: str) -> None:
        ...


class RuleBasedSelector(CampaignSelector):
    """Highest-priority eligible campaign, respecting encounter caps.

    Tie-breaker: least-recently-played first, so we rotate fairly between
    equal-priority campaigns instead of starving any of them.
    """

    def __init__(self, store: CampaignStore):
        self.store = store
        self._last_play: dict[str, float] = {}
        self._lock = threading.Lock()

    def select(self, ctx: DecisionContext) -> Campaign | None:
        now = ctx.timestamp
        candidates: list[Campaign] = []
        for c in self.store.active_campaigns():
            if not c.matches(ctx):
                continue
            last = self._last_play.get(c.campaign_id, 0.0)
            if now - last < c.encounter_cap_seconds:
                continue
            candidates.append(c)

        if not candidates:
            return None

        candidates.sort(
            key=lambda c: (-c.priority, self._last_play.get(c.campaign_id, 0.0))
        )
        chosen = candidates[0]
        log.debug(
            "selected %s (priority=%d, eligible=%d)",
            chosen.campaign_id, chosen.priority, len(candidates),
        )
        return chosen

    def record_play(self, campaign_id: str) -> None:
        with self._lock:
            self._last_play[campaign_id] = time.time()
