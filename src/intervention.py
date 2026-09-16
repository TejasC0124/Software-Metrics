from __future__ import annotations


class InterventionEngine:
    """Choose explainable prototype activities for a detected weak bridge."""

    INTERVENTIONS = {
        "semantic_phonetic": {
            "title": "Meaning to sound reinforcement",
            "focus": "Connect the word's meaning with its pronunciation.",
            "activities": [
                "Read a short sentence containing the target word and explain its meaning.",
                "Listen to the target word, repeat it, then match it to the correct picture or definition.",
                "Say the target word in a new sentence created by the learner.",
            ],
        },
        "phonetic_visual": {
            "title": "Sound to spelling reinforcement",
            "focus": "Connect phonemes with the word's letter pattern.",
            "activities": [
                "Listen to the target word and tap each phoneme in order.",
                "Highlight the letter pattern that represents the difficult sound.",
                "Choose the correctly spelled word from close alternatives, then read it aloud.",
            ],
        },
        "semantic_visual": {
            "title": "Meaning to spelling reinforcement",
            "focus": "Connect the word's meaning with its visual word form.",
            "activities": [
                "Match the target word to its definition or picture before reading it.",
                "Compare the target spelling with a visually similar distractor and identify the differing letters.",
                "Write the target word after seeing its meaning in context.",
            ],
        },
    }

    def generate_intervention(
        self, weak_mapping: str | None, target: str, response: str
    ) -> dict[str, object]:
        if weak_mapping is None:
            return {
                "title": "Positive reinforcement",
                "focus": "The response matches the target word.",
                "activities": ["Confirm the response and move to the next practice word."],
                "target": target,
                "response": response,
            }

        template = self.INTERVENTIONS.get(weak_mapping)
        if template is None:
            raise ValueError(f"Unsupported weak mapping: {weak_mapping}")
        return {
            **template,
            "target": target,
            "response": response,
            "weak_mapping": weak_mapping,
        }
