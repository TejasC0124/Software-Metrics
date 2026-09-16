from __future__ import annotations

from pathlib import Path
from typing import Any

from src.cross_modal import CrossModalEngine
from src.intervention import InterventionEngine
from src.phonetic import PhoneticEngine
from src.profile import LearnerProfileStore
from src.semantic import SemanticEngine
from src.visual import VisualEngine


class SPBEService:
    """Single application-facing entry point for a learner text attempt."""

    def __init__(self, profile_file: str | Path):
        self.cross_modal = CrossModalEngine()
        self.semantic = SemanticEngine()
        self.phonetic = PhoneticEngine()
        self.visual = VisualEngine()
        self.intervention = InterventionEngine()
        self.profile_store = LearnerProfileStore(profile_file)

    @staticmethod
    def _normalise_input(value: str, field_name: str) -> str:
        value = str(value).strip().lower()
        if not value:
            raise ValueError(f"{field_name} is required.")
        if len(value) > 60:
            raise ValueError(f"{field_name} must be 60 characters or fewer.")
        return value

    def analyse_attempt(
        self, target: str, response: str, learner_id: str
    ) -> dict[str, Any]:
        target = self._normalise_input(target, "Target word")
        response = self._normalise_input(response, "Learner response")
        learner_id = self._normalise_input(learner_id, "Learner ID")

        analysis = self.cross_modal.analyze_text_response(
            target,
            response,
            self.semantic,
            self.phonetic,
            self.visual,
        )
        intervention = self.intervention.generate_intervention(
            analysis["weak_mapping"], target, response
        )
        profile = self.profile_store.record_attempt(learner_id, analysis)
        return {
            "analysis": analysis,
            "intervention": intervention,
            "profile": profile,
            "learner_id": learner_id,
        }
