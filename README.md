# Empathy Editing for Italian Medical Question Answering

This repository contains the Streamlit dashboards and data files used for an experiment on **empathy editing in Italian medical question answering**. The pipeline has three main parts:

1. calibrate LLM judges on existing human-rated resources;
2. generate empathy-edited answers for IMB-QA;
3. evaluate the edited answers with pairwise empathy/quality judgments and factual-preservation checks.

The project is designed around a controlled LLM-as-a-judge setup. It does **not** treat the LLM as an autonomous medical answer generator. Instead, the main IMB experiment rewrites existing Italian medical answers while trying to preserve the original medical content.

## Repository contents

### Main scripts

| File | Purpose |
|---|---|
| `IDRE-calibration.py` | Streamlit dashboard for calibrating pairwise empathy judges on IDRE. |
| `AskDocs-calibration.py` | Streamlit dashboard for calibrating pairwise empathy and quality judges on the AskDocs physician--ChatGPT response data. |
| `IMB-generation-judge.py` | Streamlit dashboard for generating empathy-edited IMB-QA answers and judging them. |

### Expected data files

| File | Used by | Description |
|---|---|---|
| `IDRE.tsv` | `IDRE-calibration.py` | Italian empathy-rephrasing data. Used to test whether a judge can identify the more empathetic response. |
| `JAMA.csv` | `AskDocs-calibration.py` | AskDocs physician--ChatGPT response data from Ayers et al. Used for empathy and quality judge calibration. |
| `IMB_QA_matched_subset_1000.csv` | `IMB-generation-judge.py` | Balanced 1,000-row IMB-QA subset used for the main Italian empathy-editing experiment. |

Optional files supported by the IMB dashboard:

| File | Description |
|---|---|
| `IMB_QA_long_answers_subset_1000.csv` | Optional long-answer subset for sensitivity tests. |

## What each dashboard does

### 1. IDRE judge calibration

Run:

```bash
python -m streamlit run IDRE-calibration.py
```

This dashboard evaluates LLM judges on Italian empathy rephrasing. It uses pairwise A/B judgments:

- input: original Italian response and empathy-enhanced Italian response;
- output: judge selects `A` or `B` as more empathetic;
- gold label: the empathy-enhanced response for rows with high empathy increment.

The dashboard supports:

- balanced A/B ordering with a fixed seed;
- filtering of incomplete, broken, or language-inconsistent rows;
- mismatch inspection;
- A/B confusion matrices;
- accuracy, F1, precision, and recall.

### 2. AskDocs judge calibration

Run:

```bash
python -m streamlit run AskDocs-calibration.py
```

This dashboard evaluates LLM judges on the AskDocs physician--ChatGPT response data. It supports two judge dimensions:

- empathy preference;
- quality preference.

Gold labels are derived from average human ratings. The dashboard can filter the view to clearer cases where the average score difference is at least 1.

The dashboard supports:

- pairwise A/B judging;
- empathy/quality prompt switching;
- prompt variants;
- source-level analysis after mapping A/B choices back to physician or ChatGPT source;
- mismatch inspection;
- metrics and confusion matrices.

### 3. IMB empathy generation and judging

Run:

```bash
python -m streamlit run IMB-generation-judge.py
```

This dashboard performs the main IMB-QA experiment.

It supports:

- selecting the IMB subset variant;
- selecting generation prompt variants;
- generating empathy-edited answers;
- judging pairwise empathy preference;
- judging broad quality preference;
- checking factual preservation;
- checking unsupported medical additions;
- computing factual pass and Safe Quality Improvement (SQI);
- inspecting individual rows;
- deleting generation or judge result files when a run needs to be repeated.

The main output compares original IMB-QA answers against generated empathy-edited answers.

## Main evaluation concepts

### Pairwise empathy preference

The judge sees the patient question and two responses, A and B, and selects which response has higher empathy.

### Pairwise quality preference

The judge selects which response has higher overall quality. This is treated cautiously because quality preference may be influenced by fluency, length, or added detail.

### Factual preservation

The factual preservation check asks whether the rewritten answer preserves the clinically important information from the original answer.

### Unsupported additions

The unsupported-addition check asks whether the rewritten answer introduces new medical facts, conditions, advice, warnings, diagnoses, treatment instructions, or follow-up recommendations not present in the original answer.

### Factual pass

A generated answer passes the factual check only when:

```text
Factual pass = preservation is Yes AND unsupported additions is No
```

### Safe Quality Improvement

Safe Quality Improvement (SQI) is a conservative quality outcome:

```text
SQI = quality prefers generated answer AND factual pass is True
```

This prevents broad quality preference from being interpreted as safe improvement when the generated answer adds unsupported medical content.

## Models and API configuration

The dashboards use the OpenAI-compatible Python client. They can work with providers such as:

- OpenAI;
- Groq;
- Gemini OpenAI-compatible endpoint;
- Venice;
- other OpenAI-compatible endpoints.

Model labels used in the scripts include:

- `+ llama-3.3-70b-versatile`
- `+ openai/gpt-oss-120b`
- `+ gemini-3.1-flash-lite`
- `+ gpt-5.4-mini`
- `+ venice-uncensored`

The IMB dashboard currently focuses on:

- generator models: Llama 3.3 70B and GPT-OSS-120B;
- judge models: Llama 3.3 70B and GPT-OSS-120B.

API keys are entered in the Streamlit interface. Do not commit API keys to Git.

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install streamlit pandas numpy plotly scikit-learn openai
```

The scripts were developed for local interactive Streamlit use. If you use additional analysis or plotting scripts, you may need to install extra packages.

## Suggested workflow

1. Put the data files in the repository root:
   - `IDRE.tsv`
   - `JAMA.csv`
   - `IMB_QA_matched_subset_1000.csv`

2. Run IDRE calibration:

   ```bash
   python -m streamlit run IDRE-calibration.py
   ```

3. Run AskDocs calibration:

   ```bash
   python -m streamlit run AskDocs-calibration.py
   ```

4. Select the strongest/most consistent judges based on calibration results.

5. Run IMB generation and judging:

   ```bash
   python -m streamlit run IMB-generation-judge.py
   ```

6. Inspect:
   - empathy preference;
   - quality preference;
   - factual preservation;
   - unsupported additions;
   - factual pass;
   - SQI;
   - row-level examples.

## Output files

The dashboards save run outputs as CSV files in the working directory. Filenames include dataset, model, prompt variant, and seed information. Typical outputs include:

- generation result files;
- judge result files;
- current-run files;
- row-level metadata;
- model outputs;
- errors for failed rows.

These files are intentionally kept separate so that each generator, judge, prompt variant, and subset can be inspected independently.

## Reproducibility notes

The main experiments use:

- deterministic generation and judging where possible;
- temperature `0.0`;
- fixed A/B randomization seed, usually `42`;
- separate generation and judgment CSV files;
- row-level error logging.

For reproducibility, keep the same:

- input CSV files;
- model labels;
- provider/base URLs;
- prompt variants;
- seed;
- temperature settings.

## Data and license notes

This repository may include derived or subset data files for reproducibility. The upstream datasets have their own licenses and citation requirements. Check the original sources before redistributing modified or generated medical text.

Main upstream resources:

- IMB-QA: Italian Medical Benchmark for Question Answering.
- IDRE: Italian empathy rephrasing dataset.
- AskDocs physician--ChatGPT response dataset associated with Ayers et al.

## Citation

If you use this repository, cite the relevant upstream datasets and papers used in your experiment. At minimum, cite:

- IDRE for Italian empathy rephrasing;
- Ayers et al. for the AskDocs physician--ChatGPT response data;
- IMB-QA for the Italian medical QA benchmark.

## Notes

This code is intended for research on empathy editing and evaluation. It is not a clinical tool and should not be used for medical decision-making.
