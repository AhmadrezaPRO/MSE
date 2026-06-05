import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from openai import OpenAI
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

st.set_page_config(page_title="IDRE Empathy Judge Dashboard", page_icon="💬", layout="wide")

DEFAULT_IDRE = "IDRE.tsv"
DEFAULT_RESULTS = "current_run_+ webui_gpt-oss-20b_zero-shot_seed-42_BALANCED_AB.csv"

JUDGE_MODEL_LABELS = [
    "+ llama-3.3-70b-versatile",
    "+ llama-3.1-8b-instant",
    "+ openai/gpt-oss-120b",
    "+ webui/gpt-oss-20b",
    "+ venice-uncensored",
    "+ gemini-3.1-flash-lite",
    "+ gemini-2.5-pro",
    "+ gpt-5.4-mini",
    "+ gpt-5-mini",
    "+ gpt-4o-mini",
]

DEFAULT_SYSTEM_PROMPT = """You are an AI judge for empathy evaluation. You must choose which of two Italian sentences is more empathetic. Answer with only A or B.""".strip()

DEFAULT_USER_PROMPT = """Given these two Italian sentences, which one is more empathetic?
Answer with only A or B.

A. {sentence_a}

B. {sentence_b}
""".strip()

DEFAULT_MISMATCH_PROMPT = """You are analyzing a mismatch between an AI judge and the gold label for an empathy pairwise comparison.

The dataset row was selected because Incremento_empatia is 4 or 5, so the gold label says that ANSWER WITH EMPATHY is more empathetic than ANSWER.

User question:
{question}

ANSWER:
{normal_answer}

ANSWER WITH EMPATHY:
{empathy_answer}

Prompt order shown to the judge:
{response_order}

Judge chose:
{pred_choice}

Correct choice according to the gold label:
{gold_choice}

Briefly explain why the judge may have chosen the wrong sentence, focusing on empathy and emotional support.

Answer ONLY with valid JSON:
{{"mismatch_explanation": "brief explanation in English"}}
""".strip()


def deterministic_bool(seed: int, row_id: str) -> bool:
    digest = hashlib.md5(f"{seed}:{row_id}".encode()).hexdigest()
    return (int(digest[:8], 16) % 2) == 0


def parse_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model response")
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
    raise ValueError(f"Could not parse JSON from model output: {text[:300]}")


def normalize_bool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float, np.integer, np.floating)) and not pd.isna(x):
        return bool(int(round(float(x))))
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "sì", "si", "aumento", "increase", "increased"}:
        return True
    if s in {"false", "0", "no", "nessun aumento", "not increased", "non aumento"}:
        return False
    if re.search(r"\btrue\b", s):
        return True
    if re.search(r"\bfalse\b", s):
        return False
    raise ValueError(f"Invalid boolean label: {x}")


def extract_choice(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("Empty model response")
    # The judge is instructed to answer only A or B, but tolerate common wrappers.
    m = re.search(r"\b([AB])\b", text.upper())
    if m:
        return m.group(1)
    raise ValueError(f"Could not extract A/B choice from model output: {text[:300]}")


def extract_mismatch_explanation(raw_text: str) -> str:
    try:
        data = parse_json(raw_text)
        return str(data.get("mismatch_explanation") or data.get("reason") or "").strip()
    except Exception:
        return (raw_text or "").strip()[:500]


def parse_gold_incremento(value: Any) -> int:
    if pd.isna(value):
        raise ValueError("Missing Incremento_empatia")
    m = re.search(r"([1-5])", str(value))
    if not m:
        raise ValueError(f"Could not parse Incremento_empatia: {value}")
    score = int(m.group(1))
    return int(score >= 4)



def text_value(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def looks_incomplete(text: Any) -> bool:
    s = text_value(text)
    if len(s) < 25:
        return True
    stripped = s.rstrip()
    # Common generated-row truncation patterns: text stops after introducing a list.
    if stripped.endswith((":", ",", "come:", "ad esempio:", "tra cui:", "such as:", "include:", "including:")):
        return True
    if re.search(r"(?i)(come|ad esempio|tra cui|such as|include|including)\s*:\s*$", stripped):
        return True
    return False


def english_signal_count(text: Any) -> int:
    s = " " + text_value(text).lower() + " "
    terms = [
        " the ", " and ", " or ", " with ", " without ", " depending ", " symptoms ",
        " breast ", " cancer ", " common ", " include ", " including ", " can vary ",
        " location ", " stage ", " lump ", " underarm ", " nipple ", " redness ",
        " scaliness ", " surgery ", " after surgery", " answer ", " question ",
    ]
    return sum(1 for t in terms if t in s)


def looks_broken_row(row: pd.Series) -> bool:
    normal = text_value(row.get("Risposta", ""))
    empathy = text_value(row.get("Risposta_empatia", ""))
    if looks_incomplete(normal) or looks_incomplete(empathy):
        return True
    # If the empathy version is extremely shorter than the original, it is often truncated in this dataset.
    if len(normal) >= 120 and len(empathy) < max(60, 0.45 * len(normal)):
        return True
    # Flag obvious English-heavy rows, not isolated borrowed words.
    if english_signal_count(normal) >= 4 or english_signal_count(empathy) >= 4:
        return True
    artifact_re = r"(?i)(#\s*utente|#\s*assistente|^utente\s*:|^assistente\s*:|answer with empathy|colonna answer)"
    if re.search(artifact_re, normal) or re.search(artifact_re, empathy):
        return True
    return False


def add_broken_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    flags = []
    reasons = []
    for _, row in df.iterrows():
        row_reasons = []
        if looks_incomplete(row.get("Risposta", "")):
            row_reasons.append("original_incomplete")
        if looks_incomplete(row.get("Risposta_empatia", "")):
            row_reasons.append("empathy_incomplete")
        normal = text_value(row.get("Risposta", ""))
        empathy = text_value(row.get("Risposta_empatia", ""))
        if len(normal) >= 120 and len(empathy) < max(60, 0.45 * len(normal)):
            row_reasons.append("empathy_much_shorter_than_original")
        if english_signal_count(normal) >= 4:
            row_reasons.append("original_english_heavy")
        if english_signal_count(empathy) >= 4:
            row_reasons.append("empathy_english_heavy")
        artifact_re = r"(?i)(#\s*utente|#\s*assistente|^utente\s*:|^assistente\s*:|answer with empathy|colonna answer)"
        if re.search(artifact_re, normal) or re.search(artifact_re, empathy):
            row_reasons.append("prompt_artifact")
        flags.append(bool(row_reasons))
        reasons.append(";".join(row_reasons))
    df["broken_row_flag"] = flags
    df["broken_row_reason"] = reasons
    return df

@st.cache_data(show_spinner=False)
def load_tsv(uploaded_file, default_path: str) -> Tuple[pd.DataFrame, str]:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, sep="\t")
        return df, uploaded_file.name
    df = pd.read_csv(default_path, sep="\t")
    return df, os.path.basename(default_path)


def call_model(api_key: str, model: str, base_url: str, messages: List[Dict[str, str]], temperature: float, json_mode: bool = False) -> str:
    client_kwargs = {"api_key": api_key}
    if str(base_url).strip():
        client_kwargs["base_url"] = str(base_url).strip()
    client = OpenAI(**client_kwargs)
    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content or ""


def assign_balanced_ab_order(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Assign A/B order with an exact 50/50 gold-letter split after all filters.

    The previous version used independent per-row hashing, which is reproducible
    but not guaranteed to be exactly balanced after broken rows are removed.
    This function balances the *correct answer letter* on the final runnable set.
    """
    out = df.copy()
    if len(out) == 0:
        out["__empathy_is_a"] = []
        return out

    rng = np.random.default_rng(int(seed))
    positions = np.arange(len(out))
    rng.shuffle(positions)

    # With an even number, this is exactly 50/50. With an odd number, it differs by one.
    n_empathy_as_a = len(out) // 2
    empathy_as_a_positions = set(positions[:n_empathy_as_a].tolist())
    out["__empathy_is_a"] = [i in empathy_as_a_positions for i in range(len(out))]
    return out


def build_ab_pair(row: pd.Series, seed: int) -> Tuple[str, str, str, str]:
    normal = str(row["Risposta"])
    empathy = str(row["Risposta_empatia"])

    # Prefer the precomputed balanced assignment. Fall back to deterministic hash
    # only for safety if a row is inspected before assignment.
    if "__empathy_is_a" in row.index and pd.notna(row.get("__empathy_is_a")):
        empathy_is_a = bool(row.get("__empathy_is_a"))
    else:
        row_id = str(row["id"])
        empathy_is_a = not deterministic_bool(seed, row_id)

    if empathy_is_a:
        sentence_a = empathy
        sentence_b = normal
        order = "answer_with_empathy_A__answer_B"
        gold_choice = "A"
    else:
        sentence_a = normal
        sentence_b = empathy
        order = "answer_A__answer_with_empathy_B"
        gold_choice = "B"
    return sentence_a, sentence_b, order, gold_choice


def safe_format_prompt(template: str, **kwargs) -> str:
    protected = template
    for key in kwargs:
        protected = protected.replace("{" + key + "}", f"@@PLACEHOLDER_{key}@@")
    protected = protected.replace("{", "{{").replace("}", "}}")
    for key in kwargs:
        protected = protected.replace(f"@@PLACEHOLDER_{key}@@", "{" + key + "}")
    return protected.format(**kwargs)


def render_user_prompt(template: str, row: pd.Series, seed: int) -> Tuple[str, str, str]:
    sentence_a, sentence_b, order, gold_choice = build_ab_pair(row, seed)
    prompt = safe_format_prompt(
        template,
        question=str(row["Domanda"]),
        sentence_a=sentence_a,
        sentence_b=sentence_b,
    )
    return prompt, order, gold_choice


def judge_empathy(api_key: str, model: str, base_url: str, system_prompt: str, user_template: str, mismatch_prompt: str, row: pd.Series, seed: int, temperature: float) -> Dict[str, Any]:
    exact_user_prompt, response_order, gold_choice = render_user_prompt(user_template, row, seed)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": exact_user_prompt},
    ]
    raw = call_model(api_key, model, base_url, messages, temperature, json_mode=False)
    pred_choice = extract_choice(raw)

    pred = int(pred_choice == gold_choice)
    gold = 1  # This app evaluates only rows with Incremento_empatia 4 or 5.
    mismatch_explanation = ""
    mismatch_raw = ""

    if pred != gold:
        mismatch_text = safe_format_prompt(
            mismatch_prompt,
            question=str(row["Domanda"]),
            normal_answer=str(row["Risposta"]),
            empathy_answer=str(row["Risposta_empatia"]),
            response_order=response_order,
            pred_choice=pred_choice,
            gold_choice=gold_choice,
        )
        mismatch_raw = call_model(
            api_key,
            model,
            base_url,
            [
                {"role": "system", "content": "Answer only with valid JSON."},
                {"role": "user", "content": mismatch_text},
            ],
            temperature,
            json_mode=True,
        )
        mismatch_explanation = extract_mismatch_explanation(mismatch_raw)

    return {
        "pred_increased_empathy": pred,
        "gold_increased_empathy": gold,
        "pred_choice": pred_choice,
        "gold_choice": gold_choice,
        "is_match": int(pred_choice == gold_choice),
        "gold_incremento_empatia_raw": str(row["Incremento_empatia"]),
        "empathy_reason": f"Judge chose {pred_choice}; gold choice is {gold_choice}.",
        "mismatch_explanation": mismatch_explanation,
        "response_order": response_order,
        "exact_system_prompt": system_prompt,
        "exact_user_prompt": exact_user_prompt,
        "raw_judge_response": raw,
        "raw_mismatch_response": mismatch_raw,
        "judge_error": 0,
        "judge_error_message": "",
    }


def safe_judge_empathy(**kwargs) -> Dict[str, Any]:
    last_error = None
    for _ in range(2):
        try:
            return judge_empathy(**kwargs)
        except Exception as e:
            last_error = e
            time.sleep(1)

    row = kwargs["row"]
    seed = kwargs["seed"]
    exact_user_prompt, response_order, gold_choice = render_user_prompt(kwargs["user_template"], row, seed)
    return {
        "pred_increased_empathy": np.nan,
        "gold_increased_empathy": parse_gold_incremento(row["Incremento_empatia"]),
        "pred_choice": np.nan,
        "gold_choice": gold_choice,
        "is_match": np.nan,
        "gold_incremento_empatia_raw": str(row["Incremento_empatia"]),
        "empathy_reason": "",
        "mismatch_explanation": "",
        "response_order": response_order,
        "exact_system_prompt": kwargs["system_prompt"],
        "exact_user_prompt": exact_user_prompt,
        "raw_judge_response": "",
        "raw_mismatch_response": "",
        "judge_error": 1,
        "judge_error_message": str(last_error),
    }


def ensure_id_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "id" not in df.columns:
        if "Unnamed: 0" in df.columns:
            df["id"] = df["Unnamed: 0"].astype(str)
        else:
            df["id"] = df.index.astype(str)
    return df


def empty_results() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "id", "dataset_name", "question_text", "normal_answer_text", "empathy_answer_text",
        "judge_model", "api_model", "shot_mode", "seed", "response_order",
        "gold_increased_empathy", "pred_increased_empathy", "pred_choice", "gold_choice", "is_match", "gold_incremento_empatia_raw",
        "empathy_reason", "mismatch_explanation", "exact_system_prompt", "exact_user_prompt",
        "raw_judge_response", "raw_mismatch_response", "judge_error", "judge_error_message", "scored_at",
    ])



TEXT_RESULT_COLUMNS = [
    "id", "dataset_name", "question_text", "normal_answer_text", "empathy_answer_text",
    "judge_model", "api_model", "shot_mode", "response_order", "pred_choice", "gold_choice",
    "gold_incremento_empatia_raw", "empathy_reason", "mismatch_explanation",
    "exact_system_prompt", "exact_user_prompt", "raw_judge_response", "raw_mismatch_response",
    "judge_error_message", "scored_at",
]

NUMERIC_RESULT_COLUMNS = [
    "seed", "gold_increased_empathy", "pred_increased_empathy", "is_match", "judge_error",
]


def coerce_result_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Keep result columns assignment-safe under pandas 3/Python 3.14.

    Empty CSV columns are often inferred as float64. Assigning strings like A/B or
    an empty error message into those columns can raise:
    TypeError: Invalid value '' for dtype 'float64'.
    """
    df = df.copy()
    for col in TEXT_RESULT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("object")
            df[col] = df[col].where(~pd.isna(df[col]), "")
    for col in NUMERIC_RESULT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def ensure_results_schema(df: pd.DataFrame) -> pd.DataFrame:
    base = empty_results()
    df = df.copy()
    for col in base.columns:
        if col not in df.columns:
            df[col] = "" if col in TEXT_RESULT_COLUMNS else np.nan
    return coerce_result_dtypes(df[base.columns])


def load_results(path: str) -> pd.DataFrame:
    # Do NOT cache this. The CSV is updated during a run, and caching makes
    # Streamlit show/download an old empty/header-only dataframe after scoring.
    if not path or not os.path.exists(path):
        return empty_results()
    return recover_choices_from_raw(ensure_results_schema(pd.read_csv(path)))


def save_results(df: pd.DataFrame, path: str) -> None:
    if path and os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    ensure_results_schema(df.copy()).to_csv(path, index=False)


def make_view(test_df: pd.DataFrame, results_df: pd.DataFrame, dataset_name: str, judge_model: str, seed: int) -> pd.DataFrame:
    base_rows = []
    for _, r in test_df.iterrows():
        _, _, order, gold_choice = build_ab_pair(r, seed)
        base_rows.append({
            "id": str(r["id"]),
            "dataset_name": dataset_name,
            "question_text": str(r["Domanda"]),
            "normal_answer_text": str(r["Risposta"]),
            "empathy_answer_text": str(r["Risposta_empatia"]),
            "gold_increased_empathy": int(parse_gold_incremento(r["Incremento_empatia"])),
            "gold_choice": gold_choice,
            "response_order": order,
            "gold_incremento_empatia_raw": str(r["Incremento_empatia"]),
            "broken_row_flag": bool(r.get("broken_row_flag", False)),
            "broken_row_reason": str(r.get("broken_row_reason", "")),
        })
    base = pd.DataFrame(base_rows)
    results_df = ensure_results_schema(results_df.copy())
    subset = results_df[
        (results_df["judge_model"].astype(str) == judge_model)
        & (results_df["shot_mode"].astype(str) == "zero-shot")
        & (pd.to_numeric(results_df["seed"], errors="coerce").fillna(seed).astype(int) == int(seed))
    ].copy()
    subset["id"] = subset["id"].astype(str)
    merged = base.merge(
        subset.drop(columns=["dataset_name", "question_text", "normal_answer_text", "empathy_answer_text", "gold_increased_empathy", "gold_incremento_empatia_raw", "gold_choice", "response_order"], errors="ignore"),
        on="id",
        how="left",
    )
    merged["judge_model"] = merged.get("judge_model", judge_model).fillna(judge_model)
    merged["shot_mode"] = merged.get("shot_mode", "zero-shot").fillna("zero-shot")
    merged["seed"] = merged.get("seed", seed).fillna(seed).astype(int)
    merged["judge_error"] = pd.to_numeric(merged.get("judge_error", 0), errors="coerce").fillna(0).astype(int)
    return add_match_columns(recover_choices_from_raw(ensure_results_schema(merged)))


def upsert_view(all_results: pd.DataFrame, view_df: pd.DataFrame, judge_model: str, seed: int) -> pd.DataFrame:
    all_results = ensure_results_schema(all_results.copy())
    mask = (
        (all_results["judge_model"].astype(str) == judge_model)
        & (all_results["shot_mode"].astype(str) == "zero-shot")
        & (pd.to_numeric(all_results["seed"], errors="coerce").fillna(seed).astype(int) == int(seed))
    ) if not all_results.empty else pd.Series([], dtype=bool)
    remaining = all_results.loc[~mask].copy() if not all_results.empty else empty_results()
    return ensure_results_schema(pd.concat([remaining, view_df], ignore_index=True, sort=False))


def compute_metrics(view_df: pd.DataFrame) -> Dict[str, float]:
    done = view_df[view_df["pred_choice"].astype(str).isin(["A", "B"])].copy()
    if done.empty:
        return {}

    y_true = done["gold_choice"].astype(str).values
    y_pred = done["pred_choice"].astype(str).values
    match = y_true == y_pred

    return {
        "rows_scored": int(len(done)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=["A", "B"], average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=["A", "B"], average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=["A", "B"], average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, labels=["A", "B"], average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, labels=["A", "B"], average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=["A", "B"], average="weighted", zero_division=0)),
        "precision_A": float(precision_score(y_true, y_pred, labels=["A"], average="macro", zero_division=0)),
        "recall_A": float(recall_score(y_true, y_pred, labels=["A"], average="macro", zero_division=0)),
        "f1_A": float(f1_score(y_true, y_pred, labels=["A"], average="macro", zero_division=0)),
        "precision_B": float(precision_score(y_true, y_pred, labels=["B"], average="macro", zero_division=0)),
        "recall_B": float(recall_score(y_true, y_pred, labels=["B"], average="macro", zero_division=0)),
        "f1_B": float(f1_score(y_true, y_pred, labels=["B"], average="macro", zero_division=0)),
        "mismatches": int((~match).sum()),
    }



def recover_choices_from_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Fill pred_choice from raw_judge_response when an older result file saved the raw answer but not the parsed letter."""
    df = df.copy()
    if "raw_judge_response" not in df.columns:
        return df
    missing = ~df["pred_choice"].astype(str).isin(["A", "B"])
    for idx in df.index[missing]:
        raw = df.at[idx, "raw_judge_response"]
        try:
            choice = extract_choice(raw)
        except Exception:
            choice = ""
        if choice in ["A", "B"]:
            df.at[idx, "pred_choice"] = choice
    return df

def add_match_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "is_match" not in df.columns:
        df["is_match"] = np.nan
    scored = df["pred_choice"].astype(str).isin(["A", "B"])
    df.loc[scored, "is_match"] = (df.loc[scored, "gold_choice"].astype(str) == df.loc[scored, "pred_choice"].astype(str)).astype(int)
    df.loc[scored, "pred_increased_empathy"] = df.loc[scored, "is_match"].astype(int)
    df["gold_increased_empathy"] = 1
    return df


st.title("💬 IDRE Empathy Increase Judge")
st.caption("Zero-shot only. Uses only rows with Incremento_empatia 4–5. The judge sees A/B in a fixed randomized order and must answer only A or B. Use 'Number of rows to run now' to test only the next N pending rows.")

with st.sidebar:
    st.header("Controls")
    api_key = st.text_input("API key", type="password")
    base_url = st.text_input("Base URL (optional)", value="https://api.groq.com/openai/v1")

    model_choice = st.selectbox("Judge model label", JUDGE_MODEL_LABELS, index=3)

    model_map = {
        "+ gemini-3.1-flash-lite": "gemini-3.1-flash-lite",
        "+ gemini-2.5-pro": "gemini-2.5-pro",
        "+ gpt-4o-mini": "gpt-4o-mini",
        "+ gpt-5-mini": "gpt-5-mini",
        "+ gpt-5.4-mini": "gpt-5.4-mini",
        "+ openai/gpt-oss-120b": "openai/gpt-oss-120b",
        "+ webui/gpt-oss-20b": "unsloth_gpt-oss-20b-GGUF_gpt-oss-20b-F16.gguf",
        "+ llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
        "+ llama-3.1-8b-instant": "llama-3.1-8b-instant",
        "+ venice-uncensored": "venice-uncensored",
    }
    judge_model = model_choice
    api_model = model_map[model_choice]

    st.text_input("Mode", value="zero-shot", disabled=True)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    batch_size = st.number_input("Number of rows to run now", min_value=1, value=50, step=1, help="This limits the Run next batch button. For example, 50 means score only the next 50 pending rows.")
    exclude_broken_rows = st.checkbox("Exclude obviously broken rows", value=True, help="Filters rows where the original or empathy response looks truncated, English-heavy, or contains prompt artifacts. This is not a strict quality filter.")
    with st.expander("Advanced: prompt order reproducibility", expanded=False):
        random_seed = st.number_input("Prompt order seed", min_value=0, value=42, step=1, help="Only controls whether the empathy answer is shown as A or B. It is not a few-shot/random-shot setting.")
    on_row_error = st.selectbox("If a row fails", ["Skip and continue", "Stop run after first failure"], index=1)
    test_file = st.file_uploader("Upload IDRE TSV", type=["tsv", "txt", "csv"])
    results_path = st.text_input("Results CSV path", value=DEFAULT_RESULTS)

if not judge_model or not api_model:
    st.error("Choose or type a judge model.")
    st.stop()

try:
    test_df, test_name = load_tsv(test_file, DEFAULT_IDRE)
    test_df = ensure_id_column(test_df)
except Exception as e:
    st.error(f"Could not load IDRE TSV: {e}")
    st.stop()

required_cols = ["Domanda", "Risposta", "Risposta_empatia", "Incremento_empatia"]
missing = [c for c in required_cols if c not in test_df.columns]
if missing:
    st.error(f"IDRE TSV is missing required columns: {missing}")
    st.stop()

# Supervisor instruction for this A/B inspection version: evaluate only rows whose empathy-increase annotation is 4 or 5.
full_rows_count = len(test_df)
test_df = test_df[test_df["Incremento_empatia"].apply(parse_gold_incremento).astype(int) == 1].copy().reset_index(drop=True)
positive_rows_count = len(test_df)
test_df = add_broken_flags(test_df)
broken_rows_count = int(test_df["broken_row_flag"].sum())
if exclude_broken_rows:
    test_df = test_df[~test_df["broken_row_flag"]].copy().reset_index(drop=True)

# Important: balance A/B AFTER the 4/5 filter and AFTER broken-row exclusion.
# This guarantees Gold A / Gold B is exactly 50/50 when the final row count is even.
test_df = assign_balanced_ab_order(test_df, int(random_seed)).reset_index(drop=True)

all_results = load_results(results_path)
view_df = coerce_result_dtypes(add_match_columns(make_view(test_df, all_results, test_name, judge_model, int(random_seed))))

st.subheader("Current configuration")
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Judge label", judge_model)
c2.metric("API model", api_model)
c3.metric("Mode", "zero-shot")
c4.metric("Rows used", len(test_df))
c5.metric("Broken 4/5 rows", broken_rows_count)
c6.metric("Prompt order seed", int(random_seed))
st.caption(f"Full TSV rows: {full_rows_count} · 4/5 eligible rows: {positive_rows_count} · Broken rows detected inside 4/5 set: {broken_rows_count} · Exclude broken rows: {exclude_broken_rows}")

st.subheader("Prompt templates")
if "system_prompt" not in st.session_state:
    st.session_state["system_prompt"] = DEFAULT_SYSTEM_PROMPT
if "user_prompt" not in st.session_state:
    st.session_state["user_prompt"] = DEFAULT_USER_PROMPT
if "mismatch_prompt" not in st.session_state:
    st.session_state["mismatch_prompt"] = DEFAULT_MISMATCH_PROMPT

if st.button("Reset prompts"):
    st.session_state["system_prompt"] = DEFAULT_SYSTEM_PROMPT
    st.session_state["user_prompt"] = DEFAULT_USER_PROMPT
    st.session_state["mismatch_prompt"] = DEFAULT_MISMATCH_PROMPT

st.text_area("System prompt", key="system_prompt", height=110)
st.text_area("Empathy evaluation prompt", key="user_prompt", height=420)
st.text_area("Mismatch explanation prompt", key="mismatch_prompt", height=210)

completed_mask = view_df["pred_choice"].astype(str).isin(["A", "B"])
failed_mask = view_df["judge_error"].fillna(0).astype(int).eq(1)
metrics = compute_metrics(view_df)

st.subheader("Metrics")
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Rows scored", metrics.get("rows_scored", 0))
m2.metric("A/B accuracy", f"{metrics['accuracy']:.3f}" if metrics else "")
m3.metric("Macro F1", f"{metrics['macro_f1']:.3f}" if metrics else "")
m4.metric("Macro precision", f"{metrics['macro_precision']:.3f}" if metrics else "")
m5.metric("Macro recall", f"{metrics['macro_recall']:.3f}" if metrics else "")
m6.metric("Mismatches", metrics.get("mismatches", 0) if metrics else 0)

m7, m8, m9, m10, m11, m12 = st.columns(6)
m7.metric("Gold A rows", int((view_df["gold_choice"].astype(str) == "A").sum()))
m8.metric("Gold B rows", int((view_df["gold_choice"].astype(str) == "B").sum()))
m9.metric("Pred A", int((view_df["pred_choice"].astype(str) == "A").sum()))
m10.metric("Pred B", int((view_df["pred_choice"].astype(str) == "B").sum()))
m11.metric("Failed rows", int(failed_mask.sum()))
order_counts = view_df["response_order"].fillna("").value_counts().to_dict()
m12.metric("Prompt order split", f"Empathy=A {order_counts.get('answer_with_empathy_A__answer_B', 0)} / Empathy=B {order_counts.get('answer_A__answer_with_empathy_B', 0)}")

if metrics:
    with st.expander("Detailed precision / recall / F1", expanded=True):
        metric_table = pd.DataFrame([
            {"label": "A", "precision": metrics["precision_A"], "recall": metrics["recall_A"], "f1": metrics["f1_A"]},
            {"label": "B", "precision": metrics["precision_B"], "recall": metrics["recall_B"], "f1": metrics["f1_B"]},
            {"label": "macro avg", "precision": metrics["macro_precision"], "recall": metrics["macro_recall"], "f1": metrics["macro_f1"]},
            {"label": "weighted avg", "precision": metrics["weighted_precision"], "recall": metrics["weighted_recall"], "f1": metrics["weighted_f1"]},
        ])
        st.dataframe(metric_table, use_container_width=True, hide_index=True)
        st.caption("These metrics treat A and B as the two classes. In this 4/5-only A/B experiment, accuracy is the main score; macro F1/precision/recall are included for consistency with your previous dashboard.")

run_summary = st.session_state.get("last_run_summary")
if run_summary:
    if run_summary.get("stopped_early"):
        st.warning(f"Last run attempted {run_summary['attempted']} rows, failed {run_summary['failed']}, and stopped on row {run_summary['last_failed_id']}.")
        if run_summary.get("last_failed_error"):
            st.error(f"Failure reason: {run_summary['last_failed_error']}")
    else:
        st.info(f"Last run attempted {run_summary['attempted']} rows and failed {run_summary['failed']}.")
        if run_summary.get("last_failed_error"):
            st.error(f"Last failure reason: {run_summary['last_failed_error']}")

b1, b2, b3, b4 = st.columns(4)
selected_ids = st.multiselect("Specific IDs to run", options=view_df["id"].astype(str).tolist(), default=[])
run_selected = b1.button("▶ Run selected IDs", type="primary", use_container_width=True)
run_next = b2.button(f"▶ Run next {int(batch_size)} pending rows", use_container_width=True)
run_all = b3.button("▶ Run all remaining", use_container_width=True)
reset_current = b4.button("🗑 Reset current model/seed", use_container_width=True)

if reset_current:
    view_df = coerce_result_dtypes(add_match_columns(make_view(test_df, empty_results(), test_name, judge_model, int(random_seed))))
    all_results = upsert_view(all_results, view_df, judge_model, int(random_seed))
    save_results(all_results, results_path)
    st.success("Current model / prompt-order-seed results reset.")
    st.rerun()

progress_slot = st.empty()
status_slot = st.empty()


def run_rows(target_indices: List[int], stop_on_error: bool) -> Dict[str, Any]:
    global view_df
    view_df = coerce_result_dtypes(view_df)
    progress = progress_slot.progress(0)
    attempted = 0
    failed = 0
    stopped_early = False
    last_failed_id = None
    last_failed_error = ""

    for n, idx in enumerate(target_indices, start=1):
        row = test_df.loc[test_df["id"].astype(str) == str(view_df.loc[idx, "id"])].iloc[0]
        attempted += 1
        status_slot.info(f"Scoring row {row['id']} ({n}/{len(target_indices)})")
        pred = safe_judge_empathy(
            api_key=api_key,
            model=api_model,
            base_url=base_url,
            system_prompt=st.session_state["system_prompt"],
            user_template=st.session_state["user_prompt"],
            mismatch_prompt=st.session_state["mismatch_prompt"],
            row=row,
            seed=int(random_seed),
            temperature=float(temperature),
        )
        for key, value in pred.items():
            if key not in view_df.columns:
                view_df[key] = pd.Series([""] * len(view_df), dtype="object")
            if key in TEXT_RESULT_COLUMNS or isinstance(value, str):
                view_df[key] = view_df[key].astype("object")
            view_df.at[idx, key] = value

        view_df.at[idx, "judge_model"] = judge_model
        view_df.at[idx, "api_model"] = api_model
        view_df.at[idx, "shot_mode"] = "zero-shot"
        view_df.at[idx, "seed"] = int(random_seed)
        view_df.at[idx, "scored_at"] = pd.Timestamp.utcnow().isoformat()

        if int(pred.get("judge_error", 0)) == 1:
            failed += 1
            last_failed_id = row["id"]
            last_failed_error = str(pred.get("judge_error_message", ""))
            if stop_on_error:
                stopped_early = True
                break

        all_current = upsert_view(load_results(results_path), view_df, judge_model, int(random_seed))
        save_results(all_current, results_path)
        progress.progress(n / len(target_indices))

    status_slot.empty()
    return {
        "attempted": attempted,
        "failed": failed,
        "stopped_early": stopped_early,
        "last_failed_id": last_failed_id,
        "last_failed_error": last_failed_error,
    }

if run_selected or run_next or run_all:
    if not api_key.strip():
        st.error("Enter an API key first.")
    else:
        pending_indices = view_df.index[~view_df["pred_choice"].astype(str).isin(["A", "B"])].tolist()
        if run_selected:
            target_indices = view_df.index[view_df["id"].astype(str).isin([str(x) for x in selected_ids])].tolist()
        elif run_next:
            target_indices = pending_indices[:int(batch_size)]
        else:
            target_indices = pending_indices

        if not target_indices:
            st.warning("No rows selected to run.")
        else:
            st.session_state["last_run_summary"] = run_rows(target_indices, stop_on_error=(on_row_error == "Stop run after first failure"))
            st.rerun()

st.subheader("A/B matrix view")

done_for_cm = view_df[view_df["pred_choice"].astype(str).isin(["A", "B"])].copy()
if done_for_cm.empty:
    gold_counts = view_df["gold_choice"].astype(str).value_counts().reindex(["A", "B"], fill_value=0)
    st.info("No scored rows yet for this selected model / prompt-order-seed / results file. The file may contain only initialized rows, or the results path/model selection may not match your completed run.")
    st.write("Gold randomization check, before scoring:")
    st.dataframe(pd.DataFrame({"gold_choice": ["A", "B"], "count": [int(gold_counts.get("A", 0)), int(gold_counts.get("B", 0))]}), use_container_width=True)
else:
    choice_cm_df = pd.crosstab(
        done_for_cm["gold_choice"].astype(str),
        done_for_cm["pred_choice"].astype(str),
        rownames=["Gold choice"],
        colnames=["Pred choice"],
        dropna=False,
    ).reindex(index=["A", "B"], columns=["A", "B"], fill_value=0)

    st.markdown("**Gold A/B vs Predicted A/B**")
    # Display orientation requested by user:
    #   top-right = Gold B / Pred B
    #   bottom-left = Gold A / Pred A
    # This means rows are displayed as B first, then A, while columns stay A then B.
    choice_cm_display_df = choice_cm_df.reindex(index=["B", "A"], columns=["A", "B"], fill_value=0)
    st.dataframe(choice_cm_display_df, use_container_width=True)
    fig_choice = px.imshow(choice_cm_df, text_auto=True, aspect="auto", origin="lower", labels=dict(x="Predicted letter", y="Gold letter", color="Count"), title="A/B choice matrix")
    fig_choice.update_yaxes(categoryorder="array", categoryarray=["A", "B"])
    st.plotly_chart(fig_choice, use_container_width=True, key="choice_matrix_chart_top")

    matrix_counts = {
        "Gold A / Pred A": int(choice_cm_df.loc["A", "A"]),
        "Gold A / Pred B": int(choice_cm_df.loc["A", "B"]),
        "Gold B / Pred A": int(choice_cm_df.loc["B", "A"]),
        "Gold B / Pred B": int(choice_cm_df.loc["B", "B"]),
    }
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Gold A / Pred A", matrix_counts["Gold A / Pred A"])
    mc2.metric("Gold A / Pred B", matrix_counts["Gold A / Pred B"])
    mc3.metric("Gold B / Pred A", matrix_counts["Gold B / Pred A"])
    mc4.metric("Gold B / Pred B", matrix_counts["Gold B / Pred B"])

st.subheader("Rows preview")
preview_mode = st.radio("Show rows", ["All rows", "Pending only", "Scored only", "Failed only", "Correct only", "Mismatches only", "Gold A / Pred A", "Gold A / Pred B", "Gold B / Pred A", "Gold B / Pred B"], horizontal=True, index=0)
if preview_mode == "Pending only":
    preview_df = view_df[~view_df["pred_choice"].astype(str).isin(["A", "B"])].copy()
elif preview_mode == "Scored only":
    preview_df = view_df[completed_mask].copy()
elif preview_mode == "Failed only":
    preview_df = view_df[failed_mask].copy()
elif preview_mode == "Correct only":
    tmp = view_df[view_df["pred_choice"].astype(str).isin(["A", "B"])].copy()
    preview_df = tmp[tmp["gold_choice"].astype(str) == tmp["pred_choice"].astype(str)]
elif preview_mode == "Mismatches only":
    tmp = view_df[view_df["pred_choice"].astype(str).isin(["A", "B"])].copy()
    preview_df = tmp[tmp["gold_choice"].astype(str) != tmp["pred_choice"].astype(str)]
elif preview_mode.startswith("Gold"):
    gold_letter = preview_mode.split()[1]
    pred_letter = preview_mode.split()[-1]
    preview_df = view_df[(view_df["gold_choice"].astype(str) == gold_letter) & (view_df["pred_choice"].astype(str) == pred_letter)].copy()
else:
    preview_df = view_df.copy()

show_cols = [
    "id", "gold_choice", "pred_choice", "is_match", "gold_increased_empathy", "pred_increased_empathy", "gold_incremento_empatia_raw", "response_order",
    "empathy_reason", "mismatch_explanation", "judge_error", "judge_error_message",
    "question_text", "normal_answer_text", "empathy_answer_text", "exact_user_prompt",
]
st.dataframe(preview_df[[c for c in show_cols if c in preview_df.columns]].fillna(""), use_container_width=True, height=430)

st.subheader("Match summary")
scored_for_match = view_df[view_df["pred_choice"].astype(str).isin(["A", "B"])].copy()
if scored_for_match.empty:
    st.info("No scored rows yet.")
else:
    scored_for_match["choice_match"] = scored_for_match["gold_choice"].astype(str) == scored_for_match["pred_choice"].astype(str)
    sm1, sm2, sm3 = st.columns(3)
    sm1.metric("Scored A/B rows", len(scored_for_match))
    sm2.metric("Correct A/B choices", int(scored_for_match["choice_match"].sum()))
    sm3.metric("Wrong A/B choices", int((~scored_for_match["choice_match"]).sum()))

st.subheader("Inspect row")
inspect_filter = st.selectbox("Inspect subset", ["All", "Pending only", "Scored only", "Failed only", "Correct only", "Mismatches only", "Gold A / Pred A", "Gold A / Pred B", "Gold B / Pred A", "Gold B / Pred B"], index=0)
inspect_source_df = view_df.copy()
inspect_source_df["_pred_num"] = pd.to_numeric(inspect_source_df["pred_increased_empathy"], errors="coerce")
inspect_source_df["_gold_num"] = pd.to_numeric(inspect_source_df["gold_increased_empathy"], errors="coerce")
if inspect_filter == "Pending only":
    inspect_source_df = inspect_source_df[~inspect_source_df["pred_choice"].astype(str).isin(["A", "B"])]
elif inspect_filter == "Scored only":
    inspect_source_df = inspect_source_df[inspect_source_df["pred_choice"].astype(str).isin(["A", "B"])]
elif inspect_filter == "Failed only":
    inspect_source_df = inspect_source_df[inspect_source_df["judge_error"].fillna(0).astype(int).eq(1)]
elif inspect_filter == "Correct only":
    inspect_source_df = inspect_source_df[inspect_source_df["pred_choice"].astype(str).isin(["A", "B"]) & (inspect_source_df["gold_choice"].astype(str) == inspect_source_df["pred_choice"].astype(str))]
elif inspect_filter == "Mismatches only":
    inspect_source_df = inspect_source_df[inspect_source_df["pred_choice"].astype(str).isin(["A", "B"]) & (inspect_source_df["gold_choice"].astype(str) != inspect_source_df["pred_choice"].astype(str))]
elif inspect_filter.startswith("Gold"):
    gold_letter = inspect_filter.split()[1]
    pred_letter = inspect_filter.split()[-1]
    inspect_source_df = inspect_source_df[(inspect_source_df["gold_choice"].astype(str) == gold_letter) & (inspect_source_df["pred_choice"].astype(str) == pred_letter)]

inspect_ids = inspect_source_df["id"].astype(str).tolist()
if inspect_ids:
    inspect_id = st.selectbox("Choose ID", options=inspect_ids, key=f"inspect_id_{judge_model}_{random_seed}")
    row = inspect_source_df[inspect_source_df["id"].astype(str) == str(inspect_id)].iloc[0]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Domanda**")
        st.write(row["question_text"])
        st.markdown("**Frase originale / Risposta**")
        st.write(row["normal_answer_text"])
    with c2:
        st.markdown("**Frase rephrasing / Risposta_empatia**")
        st.write(row["empathy_answer_text"])

    pred_value = row.get("pred_increased_empathy")
    has_pred = pd.notna(pred_value)
    pred_int = int(float(pred_value)) if has_pred else None
    gold_int = int(row["gold_increased_empathy"])
    mismatch = has_pred and (gold_int != pred_int)
    bg = "#f6caca" if mismatch else "#cfead6"
    border = "#9f1d1d" if mismatch else "#1f6b3a"
    st.markdown(
        f"""
        <div style="padding:10px 12px;border-radius:10px;border:2px solid {border};background:{bg};margin-bottom:8px;color:#111111;">
            <div style="font-size:0.9rem;font-weight:700;color:#111111;">Pairwise choice match</div>
            <div style="display:flex;gap:24px;margin-top:6px;flex-wrap:wrap;color:#111111;">
                <div><strong>Gold letter:</strong> {row.get('gold_choice', '')}</div>
                <div><strong>Pred letter:</strong> {row.get('pred_choice', '') if has_pred else ''}</div>
                <div><strong>Gold chooses empathy answer:</strong> {bool(gold_int)}</div>
                <div><strong>Pred chose empathy answer:</strong> {bool(pred_int) if has_pred else ''}</div>
                <div><strong>Incremento_empatia:</strong> {row.get('gold_incremento_empatia_raw', '')}</div>
                <div><strong>Match:</strong> {'No' if mismatch else 'Yes' if has_pred else ''}</div>
                <div><strong>Order:</strong> {row.get('response_order', '')}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    j1, j2 = st.columns(2)
    with j1:
        st.markdown("**Reasons / error details**")
        detail_info = {
            "empathy_reason": row.get("empathy_reason", ""),
            "mismatch_explanation": row.get("mismatch_explanation", ""),
            "judge_error": int(row.get("judge_error", 0)) if pd.notna(row.get("judge_error", 0)) else 0,
            "judge_error_message": row.get("judge_error_message", ""),
        }
        detail_info = {k: ("" if pd.isna(v) else v) for k, v in detail_info.items()}
        detail_info = {k: v for k, v in detail_info.items() if not ((isinstance(v, str) and str(v).strip() in {"", "nan", "None"}) or (k == "judge_error" and int(v) == 0))}
        st.json(detail_info)
    with j2:
        with st.expander("Exact system prompt", expanded=False):
            st.text("" if pd.isna(row.get("exact_system_prompt", "")) else str(row.get("exact_system_prompt", "")))
        with st.expander("Exact user prompt", expanded=False):
            st.text("" if pd.isna(row.get("exact_user_prompt", "")) else str(row.get("exact_user_prompt", "")))
        with st.expander("Raw judge JSON", expanded=False):
            st.text("" if pd.isna(row.get("raw_judge_response", "")) else str(row.get("raw_judge_response", "")))
else:
    st.info("No rows available in this subset.")

current_run_download_df = ensure_results_schema(view_df.copy())
st.download_button(
    "Download current run CSV",
    data=current_run_download_df.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"current_run_{judge_model}_zero-shot_seed-{int(random_seed)}.csv".replace("/", "_"),
    mime="text/csv",
)
st.download_button(
    "Download results CSV",
    data=load_results(results_path).to_csv(index=False).encode("utf-8-sig"),
    file_name=os.path.basename(results_path),
    mime="text/csv",
)
