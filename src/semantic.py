from __future__ import annotations

from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer


class SemanticEngine:
    """Create lexical-semantic representations and optional embeddings.

    Embedding loading is lazy so the text-response prototype can run without
    downloading a transformer model during a review demonstration.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model: SentenceTransformer | None = None

    def get_synsets(self, word: str) -> list[object]:
        try:
            return wn.synsets(str(word).lower())
        except LookupError:
            return []

    def get_semantic_information(self, word: str) -> dict[str, object]:
        synsets = self.get_synsets(word)
        definitions: list[str] = []
        examples: list[str] = []
        synonyms: set[str] = set()
        for synset in synsets:
            definitions.append(synset.definition())
            examples.extend(synset.examples())
            synonyms.update(lemma.name() for lemma in synset.lemmas())
        return {
            "word": word,
            "synsets": [synset.name() for synset in synsets],
            "definitions": definitions,
            "examples": examples,
            "synonyms": sorted(synonyms),
        }

    def get_semantic_text(self, word: str) -> str | None:
        definitions = self.get_semantic_information(word)["definitions"]
        return " ".join(definitions) if definitions else None

    def _load_embedding_model(self) -> SentenceTransformer:
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def get_embedding(self, word: str):
        semantic_text = self.get_semantic_text(word)
        if semantic_text is None:
            return None
        return self._load_embedding_model().encode(semantic_text)

    def semantic_similarity(self, target: str, response: str) -> float:
        """Return a WordNet lexical-semantic baseline in the range 0 to 1.

        It returns 0 for unrelated or unknown words. A trained sentence-
        embedding comparison remains a future enhancement.
        """
        target = str(target).strip().lower()
        response = str(response).strip().lower()
        if target == response:
            return 1.0
        target_synsets = self.get_synsets(target)
        response_synsets = self.get_synsets(response)
        similarities = [
            similarity
            for target_synset in target_synsets
            for response_synset in response_synsets
            for similarity in [target_synset.path_similarity(response_synset)]
            if similarity is not None
        ]
        return round(max(similarities), 4) if similarities else 0.0

    def process_word(self, word: str, include_embedding: bool = False) -> dict[str, object]:
        information = self.get_semantic_information(word)
        information["embedding"] = self.get_embedding(word) if include_embedding else None
        return information
