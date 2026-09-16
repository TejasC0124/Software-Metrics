from __future__ import annotations

from statistics import mean
from typing import Any


class CrossModalEngine:
    """Attribute a text-response error to a linguistic bridge.

    This is an interpretable prototype. It compares semantic, phonetic, and
    orthographic scores, then evaluates the three bridges between them. It is
    not a clinical diagnostic model.
    """

    BRIDGES = {
        "semantic_phonetic": ("semantic", "phonetic"),
        "phonetic_visual": ("phonetic", "visual"),
        "semantic_visual": ("semantic", "visual"),
    }

    @staticmethod
    def _normalise_score(score: float | None) -> float | None:
        if score is None:
            return None
        try:
            return max(0.0, min(1.0, float(score)))
        except (TypeError, ValueError):
            return None

    def calculate_cross_modal_scores(
        self,
        semantic_similarity: float | None,
        phonetic_similarity: float | None,
        visual_similarity: float | None,
    ) -> dict[str, float | None]:
        """Return normalised scores for the three representations."""
        return {
            "semantic": self._normalise_score(semantic_similarity),
            "phonetic": self._normalise_score(phonetic_similarity),
            "visual": self._normalise_score(visual_similarity),
        }

    def calculate_bridge_scores(
        self, modality_scores: dict[str, float | None]
    ) -> dict[str, float | None]:
        """Score each connection by the mean of its two endpoint scores."""
        bridge_scores: dict[str, float | None] = {}
        for bridge, endpoints in self.BRIDGES.items():
            values = [modality_scores[endpoint] for endpoint in endpoints]
            bridge_scores[bridge] = round(mean(values), 4) if None not in values else None
        return bridge_scores

    @staticmethod
    def orthographic_similarity(target: str, response: str) -> float:
        """Compatibility helper for character-level spelling similarity."""
        from difflib import SequenceMatcher

        return SequenceMatcher(None, target.lower(), response.lower()).ratio()

    @staticmethod
    def identify_weak_mapping(bridge_scores: dict[str, float | None]) -> str | None:
        valid_scores = {
            bridge: score
            for bridge, score in bridge_scores.items()
            if score is not None
        }
        return min(valid_scores, key=valid_scores.get) if valid_scores else None

    def analyze_error(
        self,
        target: str,
        response: str,
        semantic_similarity: float | None,
        phonetic_similarity: float | None,
        visual_similarity: float | None,
    ) -> dict[str, Any]:
        """Analyse a response when modality scores are already available."""
        target = str(target).strip().lower()
        response = str(response).strip().lower()
        is_correct = target == response
        modality_scores = self.calculate_cross_modal_scores(
            semantic_similarity, phonetic_similarity, visual_similarity
        )
        bridge_scores = self.calculate_bridge_scores(modality_scores)
        weak_mapping = None if is_correct else self.identify_weak_mapping(bridge_scores)

        valid_bridges = sorted(
            score for score in bridge_scores.values() if score is not None
        )
        confidence = (
            round(valid_bridges[1] - valid_bridges[0], 4)
            if len(valid_bridges) > 1 and weak_mapping is not None
            else 0.0
        )

        return {
            "target": target,
            "response": response,
            "is_correct": is_correct,
            "modality_scores": modality_scores,
            "bridge_scores": bridge_scores,
            "weak_mapping": weak_mapping,
            "attribution_confidence": confidence,
            "orthographic_similarity": modality_scores["visual"],
        }

    def analyze_text_response(
        self,
        target: str,
        response: str,
        semantic_engine: Any,
        phonetic_engine: Any,
        visual_engine: Any,
    ) -> dict[str, Any]:
        """Calculate all available text scores and attribute the error."""
        semantic_score = semantic_engine.semantic_similarity(target, response)
        phonetic_score = phonetic_engine.phonetic_similarity(target, response)
        visual_score = visual_engine.spelling_similarity(target, response)
        return self.analyze_error(
            target,
            response,
            semantic_score,
            phonetic_score,
            visual_score,
        )
