from __future__ import annotations

import json
import threading
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class LearnerProfileStore:
    """Persist a small prototype Cross-Modal Confusion Profile as JSON."""

    DEFAULT_BRIDGES = {
        "semantic_phonetic": {"strength": 0.5, "observations": 0, "weak_events": 0},
        "phonetic_visual": {"strength": 0.5, "observations": 0, "weak_events": 0},
        "semantic_visual": {"strength": 0.5, "observations": 0, "weak_events": 0},
    }

    def __init__(self, profile_file: str | Path):
        self.profile_file = Path(profile_file)
        # The web app handles requests on several threads, and each attempt
        # rewrites the whole JSON file, so writes must be serialised.
        self._lock = threading.Lock()

    def _load_all(self) -> dict[str, Any]:
        if not self.profile_file.exists():
            return {"profiles": {}}
        try:
            with self.profile_file.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data.get("profiles"), dict) else {"profiles": {}}
        except (OSError, json.JSONDecodeError, AttributeError):
            return {"profiles": {}}

    def _save_all(self, data: dict[str, Any]) -> None:
        self.profile_file.parent.mkdir(parents=True, exist_ok=True)
        temporary_file = self.profile_file.with_suffix(".tmp")
        with temporary_file.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
        temporary_file.replace(self.profile_file)

    def record_attempt(self, learner_id: str, analysis: dict[str, Any]) -> dict[str, Any]:
        learner_id = str(learner_id).strip()
        if not learner_id:
            raise ValueError("learner_id must not be empty")
        with self._lock:
            return self._record_attempt(learner_id, analysis)

    def _record_attempt(self, learner_id: str, analysis: dict[str, Any]) -> dict[str, Any]:
        all_profiles = self._load_all()
        profiles = all_profiles["profiles"]
        profile = profiles.setdefault(
            learner_id,
            {
                "attempts": 0,
                "bridges": deepcopy(self.DEFAULT_BRIDGES),
                "recent_attempts": [],
            },
        )

        profile["attempts"] += 1
        for bridge, score in analysis["bridge_scores"].items():
            if score is None:
                continue
            bridge_profile = profile["bridges"][bridge]
            observations = bridge_profile["observations"]
            bridge_profile["strength"] = round(
                (bridge_profile["strength"] * observations + float(score))
                / (observations + 1),
                4,
            )
            bridge_profile["observations"] = observations + 1

        weak_mapping = analysis.get("weak_mapping")
        if weak_mapping:
            profile["bridges"][weak_mapping]["weak_events"] += 1

        profile["recent_attempts"].append(
            {
                "target": analysis["target"],
                "response": analysis["response"],
                "weak_mapping": weak_mapping,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        profile["recent_attempts"] = profile["recent_attempts"][-20:]
        self._save_all(all_profiles)
        return deepcopy(profile)
