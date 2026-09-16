SPBE is an early prototype for personalised dyslexia intervention. It analyses a
target word and a typed learner response across semantic, phonetic, and visual
orthographic representations, identifies the weakest bridge, selects prototype
activities, and updates a local learner profile.

Run the review demonstration:

```powershell
.\.venv\Scripts\python.exe main.py --target said --response side --learner demo_learner
```

Run the browser-based review demo, then open `http://127.0.0.1:8000`:

```powershell
.\.venv\Scripts\python.exe app.py
```

Run the standard-library test suite:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
```

Current scope: text-response prototype, dataset preprocessing, and handwriting
image preparation. Speech assessment, a trained visual encoder, a user-facing
application, clinical validation, and trained prediction models remain future work.
