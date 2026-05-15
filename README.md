# Medical Risk Judge Dashboard

This repository contains a Streamlit dashboard for evaluating whether medical chatbot answers contain **medical risk**.

## Main file

Use this file:

- `main.py`

## What the app does

The app:

1. loads a labeled medical-risk dataset
2. builds a prompt for a chosen LLM judge
3. optionally adds few-shot examples
4. supports random or BERTScore-based example retrieval
5. sends requests through an **OpenAI-compatible Python client**
6. stores predictions, mismatch explanations, and metadata
7. shows metrics, confusion matrix, row inspection, and CSV downloads

The task is:

- input: patient question + chatbot answer
- output: whether the **answer itself** is medically risky (`0` or `1`)

## Requirements

Create and activate a virtual environment, then install:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install streamlit pandas numpy plotly scikit-learn openai bert-score torch transformers sentencepiece
```

Depending on your machine, installing `torch` may differ.

## Expected data

The dashboard expects local CSV files for the medical-risk datasets.

Typical columns include:

- `id`
- `medicalRisk`
- question column
- answer column

Optional columns for note-based analysis:

- `medicalRiskNote_en`
- `medicalRiskNote_ja`
- `medicalRiskNote_de`

## How to run

```bash
python -m streamlit run main.py
```

Then open the local Streamlit URL printed in the terminal.

## Basic workflow

### 1. Choose provider/model
In the sidebar:
- enter API key
- choose provider/base URL
- choose model

The app uses an OpenAI-compatible client, so it can work with:
- OpenAI
- Groq
- Gemini OpenAI-compatible endpoint
- Venice
- other compatible providers

### 2. Choose dataset and language
Select:
- dataset (`JP` or `DE`)
- language view
- prompting mode
- note mode (`on` or `off`)

### 3. Choose prompting mode
Available modes:
- `zero-shot`
- `two-shot-random`
- `two-shot-bertscore`
- `three-shot-bertscore-2pos1neg`
- `five-shot-bertscore-3pos2neg`

### 4. Run scoring
Use the run controls to score rows.

### 5. Inspect results
The dashboard shows:
- metrics
- confusion matrix
- mode comparison table
- row-level inspection
- exact prompts used
- CSV download buttons

## Note mode

`Medical risk note in examples` controls whether few-shot examples include the gold note text:

- `on`: include note when available
- `off`: exclude note

This lets you compare whether note text improves judging performance.

## Mismatch explanation

When prediction and gold label differ, the app runs a second prompt that explains the mismatch.

If the gold label is positive and a gold note exists:
- the mismatch explainer can use that note
- for English analysis, Japanese notes can be translated when English is missing

## Output files

The app persists results to CSV and also allows:
- full results download
- current-run download

Saved outputs include:
- predicted label
- optional medical reason
- mismatch explanation
- selected example IDs
- exact prompts
- timestamps
- note-mode metadata

## Suggested git contents

Commit:
- the commented Python file
- this README
- optionally a sample results CSV without secrets

Do **not** commit API keys.
