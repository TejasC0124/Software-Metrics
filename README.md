# SPBE: Semantic-Phonetic Bridge Engine

SPBE is a prototype for personalised dyslexia intervention. Instead of only
marking a learner's typed response right or wrong, it compares the target word
with the response across three representations:

- **Semantic**: meaning, scored with WordNet path similarity.
- **Phonetic**: pronunciation, scored with CMU Pronouncing Dictionary phoneme similarity.
- **Visual**: spelling, scored with character-level similarity.

It then scores the three *bridges* between these representations, picks the
weakest one, recommends a matching activity set, and updates a per-learner
Cross-Modal Confusion Profile saved as JSON.

This is an educational prototype. It is not a clinical diagnostic tool.

## Project layout

| Path | Purpose |
| --- | --- |
| `src/semantic.py` | WordNet semantic representation and similarity |
| `src/phonetic.py` | CMU dictionary phoneme representation and similarity |
| `src/visual.py` | Spelling similarity and handwriting image validation |
| `src/cross_modal.py` | Combines the three scores and attributes the error to a bridge |
| `src/intervention.py` | Maps each weak bridge to prototype activities |
| `src/profile.py` | Thread-safe JSON learner profile store |
| `src/service.py` | Single entry point used by the CLI and the web demo |
| `src/preprocess.py` | Cleans the Dyt desktop/tablet behavioural dataset and creates train/validation/test splits |
| `src/visual_preprocess.py` | Scans the handwriting image dataset and writes a metadata CSV |
| `main.py` | Command-line demo |
| `app.py` | Local browser demo (standard library HTTP server) |
| `setup_resources.py` | Downloads WordNet and checks the dictionaries load |
| `tests/` | Unit and integration tests (standard `unittest`) |
| `outputs/` | Learner profile JSON written by the demos |

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe setup_resources.py
```

`setup_resources.py` downloads WordNet once. Without it the semantic engine
raises a clear error instead of silently scoring every word as 0.
`sentence-transformers` and `torch` are only needed for the optional embedding
comparison; the prototype runs without loading them.

## Run

Command-line attribution for one attempt:

```powershell
.\.venv\Scripts\python.exe main.py --target said --response side --learner demo_learner
```

Browser demo, then open `http://127.0.0.1:8000`:

```powershell
.\.venv\Scripts\python.exe app.py
```

Tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
```

Preprocessing needs the raw datasets, which are not included in this repository.
Place `Dyt-desktop.csv` and `Dyt-tablet.csv` in `data/raw/`, and the handwriting
images in `data/raw/handwriting/extracted/Gambo/`, then run:

```powershell
.\.venv\Scripts\python.exe -m src.preprocess
.\.venv\Scripts\python.exe -m src.visual_preprocess
```

## How attribution works

1. Each score is in the range 0 to 1 (1 means the representations agree).
2. A bridge score is the mean of its two endpoint scores:
   `semantic_phonetic`, `phonetic_visual`, `semantic_visual`.
3. The bridge with the lowest score is reported as the weak bridge. Its
   `attribution_confidence` is the gap between the two lowest bridges.
4. A correct response has no weak bridge and receives positive reinforcement.

## Known limitations

These are documented rather than hidden. Some are design questions for the
team to decide.

- **Bridge scoring can point at the wrong connection.** Because a bridge is the
  mean of its endpoints, the reported bridge is always the pair that excludes
  the *strongest* modality. Examples from the current code:
  - `bear → beer` (a sound-alike error) is reported as *semantic–phonetic*,
    because WordNet scores both words as unrelated (0.09). The phonetic and
    visual scores are high, which is the real pattern here.
  - `plant → pant` is reported as *semantic–phonetic* with confidence 0.0 (a tie).
  - `said → say` is reported as *phonetic–visual* even though the meanings are
    identical (WordNet 1.0).

  A better attribution would compare each modality with the others, or treat
  the modality that falls furthest below the learner's own baseline as the
  weak link. Changing this changes what the system recommends, so it should be
  agreed on before it is changed.
- **Semantic scores are only meaningful for real words.** Words missing from
  WordNet score 0.0, so a misspelling of a non-word looks like a meaning error.
- **Visual scoring is spelling similarity only.** The handwriting pipeline
  validates and prepares images but does not yet produce a visual score.
- **Phonetic scoring uses the first CMU pronunciation** and compares phoneme
  sequences with `difflib`. Homophones and multiple pronunciations are not
  handled.
- **Profile strengths are simple running averages** of bridge scores, not a
  trained model.

## Not yet implemented (future work)

- Speech assessment (spoken responses and pronunciation patterns).
- A trained visual encoder for handwriting.
- Trained prediction models and evaluation on the Dyt and handwriting datasets.
- Clinical validation with specialists and learners.
- A learner-facing application beyond the local demo.
