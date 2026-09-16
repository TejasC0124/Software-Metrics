import sys
from pathlib import Path

# Allow Python to find the src directory
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from semantic import SemanticEngine


def main():
    engine = SemanticEngine()

    word = "said"

    result = engine.process_word(word)

    print("Word:", result["word"])
    print("Number of synsets:", len(result["synsets"]))
    print("Definitions:")

    for definition in result["definitions"]:
        print("-", definition)

    print("\nEmbedding shape:", result["embedding"].shape)


if __name__ == "__main__":
    main()