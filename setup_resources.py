"""Download the language resources SPBE needs and check that they load.

Run once after installing requirements.txt:

    python setup_resources.py
"""

import nltk


RESOURCES = ["wordnet"]


def main() -> None:
    for resource in RESOURCES:
        print(f"Downloading NLTK resource: {resource}")
        if not nltk.download(resource, quiet=True):
            raise SystemExit(f"Could not download NLTK resource '{resource}'.")

    from nltk.corpus import wordnet

    count = len(wordnet.synsets("said"))
    if count == 0:
        raise SystemExit("WordNet installed but returned no synsets for 'said'.")

    import cmudict

    if not cmudict.dict().get("said"):
        raise SystemExit("CMU Pronouncing Dictionary did not load.")

    print(f"WordNet OK ({count} synsets for 'said').")
    print("CMU Pronouncing Dictionary OK.")
    print("Setup complete. You can now run the tests and the demo.")


if __name__ == "__main__":
    main()
