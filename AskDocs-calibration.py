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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

st.set_page_config(page_title="JAMA Pairwise Judge Dashboard", page_icon="⚕️", layout="wide")

DEFAULT_JAMA = "JAMA.csv"
DEFAULT_RESULTS = "current_run_jama_PAIRWISE_FRESH_STABLE_AB_VIEW_ONLY_seed-42.csv"

JUDGE_MODEL_LABELS = [
    "+ llama-3.3-70b-versatile",
    "+ llama-3.1-8b-instant",
    "+ openai/gpt-oss-120b",
    "+ deepseek-v4-flash",
    "+ deepseek-v4-pro",
    "+ venice-uncensored",
    "+ gemini-3.1-flash-lite",
    "+ gpt-5.4-mini",
]

PROMPT_VARIANTS = [
    "paper_minimal",
    "paper_guided_editprompt",
]

PROMPT_VARIANT_LABELS = {
    "paper_minimal": "paper_minimal: short JAMA wording",
    "paper_guided_editprompt": "paper_guided_editprompt: JAMA wording + anti-length/source-bias instruction",
}


def get_default_prompts(score_dimension: str, prompt_variant: str = "paper_minimal") -> Tuple[str, str, str]:
    dim = "quality" if str(score_dimension).lower() == "quality" else "empathy"
    variant = prompt_variant if prompt_variant in PROMPT_VARIANTS else "paper_minimal"

    system_prompt = """You are an AI judge. Follow the user instruction exactly and answer with only A or B.""".strip()

    if variant == "paper_guided_editprompt":
        if dim == "quality":
            user_prompt = """Read the entire patient question and both anonymized responses.
Which response provides higher quality of information?
Do not choose based on response length alone or assumed source. Focus on the quality of information provided.
Answer with only A or B.

Patient question:
{question}

A. {sentence_a}

B. {sentence_b}
""".strip()
            mismatch_prompt = """You are analyzing a mismatch between an AI judge and the gold label for a JAMA response quality comparison.

Gold label construction:
The gold label is based on the average of the three human evaluator quality scores. The response with the higher average quality score is treated as the preferred response.

Judge prompt variant:
paper_guided_editprompt. The judge was asked to choose the response with higher quality of information, while not choosing based on response length alone or assumed source.

Patient question:
{question}

Physician response:
{physician_answer}

ChatGPT response:
{chatgpt_answer}

Average physician quality score:
{physician_avg}

Average ChatGPT quality score:
{chatgpt_avg}

Prompt order shown to the judge:
{response_order}

Judge chose:
{pred_choice}

Correct choice according to average human quality score:
{gold_choice}

Briefly explain why the judge may have chosen the wrong response for response quality.

Answer ONLY with valid JSON:
{{"mismatch_explanation": "brief explanation in English"}}
""".strip()
        else:
            user_prompt = """Read the entire patient question and both anonymized responses.
Which response provides higher empathy or bedside manner?
Do not choose based on response length alone or assumed source. Focus on the empathy or bedside manner provided.
Answer with only A or B.

Patient question:
{question}

A. {sentence_a}

B. {sentence_b}
""".strip()
            mismatch_prompt = """You are analyzing a mismatch between an AI judge and the gold label for a JAMA response empathy comparison.

Gold label construction:
The gold label is based on the average of the three human evaluator empathy scores. The response with the higher average empathy score is treated as the preferred response.

Judge prompt variant:
paper_guided_editprompt. The judge was asked to choose the response with higher empathy or bedside manner, while not choosing based on response length alone or assumed source.

Patient question:
{question}

Physician response:
{physician_answer}

ChatGPT response:
{chatgpt_answer}

Average physician empathy score:
{physician_avg}

Average ChatGPT empathy score:
{chatgpt_avg}

Prompt order shown to the judge:
{response_order}

Judge chose:
{pred_choice}

Correct choice according to average human empathy score:
{gold_choice}

Briefly explain why the judge may have chosen the wrong response for response empathy.

Answer ONLY with valid JSON:
{{"mismatch_explanation": "brief explanation in English"}}
""".strip()
    else:
        if dim == "quality":
            user_prompt = """Given a patient question and two responses, which response has higher quality?
Answer with only A or B.

Patient question:
{question}

A. {sentence_a}

B. {sentence_b}
""".strip()
            mismatch_prompt = """You are analyzing a mismatch between an AI judge and the gold label for a JAMA response quality comparison.

Gold label construction:
The gold label is based on the average of the three human evaluator quality scores. The response with the higher average quality score is treated as the preferred response.

Judge prompt variant:
paper_minimal.

Patient question:
{question}

Physician response:
{physician_answer}

ChatGPT response:
{chatgpt_answer}

Average physician quality score:
{physician_avg}

Average ChatGPT quality score:
{chatgpt_avg}

Prompt order shown to the judge:
{response_order}

Judge chose:
{pred_choice}

Correct choice according to average human quality score:
{gold_choice}

Briefly explain why the judge may have chosen the wrong response for response quality.

Answer ONLY with valid JSON:
{{"mismatch_explanation": "brief explanation in English"}}
""".strip()
        else:
            user_prompt = """Given a patient question and two responses, which response has higher empathy?
Answer with only A or B.

Patient question:
{question}

A. {sentence_a}

B. {sentence_b}
""".strip()
            mismatch_prompt = """You are analyzing a mismatch between an AI judge and the gold label for a JAMA response empathy comparison.

Gold label construction:
The gold label is based on the average of the three human evaluator empathy scores. The response with the higher average empathy score is treated as the preferred response.

Judge prompt variant:
paper_minimal.

Patient question:
{question}

Physician response:
{physician_answer}

ChatGPT response:
{chatgpt_answer}

Average physician empathy score:
{physician_avg}

Average ChatGPT empathy score:
{chatgpt_avg}

Prompt order shown to the judge:
{response_order}

Judge chose:
{pred_choice}

Correct choice according to average human empathy score:
{gold_choice}

Briefly explain why the judge may have chosen the wrong response for response empathy.

Answer ONLY with valid JSON:
{{"mismatch_explanation": "brief explanation in English"}}
""".strip()

    return system_prompt, user_prompt, mismatch_prompt


DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT, DEFAULT_MISMATCH_PROMPT = get_default_prompts("Empathy", "paper_minimal")


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
def load_jama_csv(uploaded_file, default_path: str) -> Tuple[pd.DataFrame, str]:
    """Load the JAMA CSV. The public file is often Latin-1 encoded."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="latin1")
        return df, uploaded_file.name
    try:
        df = pd.read_csv(default_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(default_path, encoding="latin1")
    return df, os.path.basename(default_path)


def mean_score(row: pd.Series, cols: List[str]) -> float:
    vals = pd.to_numeric(pd.Series([row.get(c, np.nan) for c in cols]), errors="coerce")
    return float(vals.mean())


EMPATHY_PHYSICIAN_COLS = [
    "Eval 1 Empathy (Physician)",
    "Eval 2 Empathy (Physician)",
    "Eval 3 Empathy (Physician)",
]
EMPATHY_CHATGPT_COLS = [
    "Eval 1 Empathy (ChatGPT)",
    "Eval 2 Empathy (ChatGPT)",
    "Eval 3 Empathy (ChatGPT)",
]
QUALITY_PHYSICIAN_COLS = [
    "Eval 1 Quality (Physician)",
    "Eval 2 Quality (Physician)",
    "Eval 3 Quality (Physician)",
]
QUALITY_CHATGPT_COLS = [
    "Eval 1 Quality (ChatGPT)",
    "Eval 2 Quality (ChatGPT)",
    "Eval 3 Quality (ChatGPT)",
]


def add_average_scores(df: pd.DataFrame, score_dimension: str) -> pd.DataFrame:
    out = df.copy()
    if score_dimension == "Quality":
        p_cols, c_cols = QUALITY_PHYSICIAN_COLS, QUALITY_CHATGPT_COLS
    else:
        p_cols, c_cols = EMPATHY_PHYSICIAN_COLS, EMPATHY_CHATGPT_COLS

    out["physician_avg_score"] = out.apply(lambda r: mean_score(r, p_cols), axis=1)
    out["chatgpt_avg_score"] = out.apply(lambda r: mean_score(r, c_cols), axis=1)
    out["avg_score_diff_abs"] = (out["chatgpt_avg_score"] - out["physician_avg_score"]).abs()
    out["gold_source"] = np.where(out["chatgpt_avg_score"] > out["physician_avg_score"], "ChatGPT",
                                  np.where(out["physician_avg_score"] > out["chatgpt_avg_score"], "Physician", "Tie"))
    return out



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

    This balances the correct answer letter on the final runnable set.
    """
    out = df.copy()
    if len(out) == 0:
        out["__gold_is_a"] = []
        return out

    keys = []
    for _, row in out.iterrows():
        row_id = str(row["id"])
        digest = hashlib.md5(f"{seed}:{row_id}".encode()).hexdigest()
        keys.append(int(digest[:12], 16))
    out["__shuffle_key"] = keys
    out = out.sort_values("__shuffle_key").reset_index(drop=True)

    n_gold_a = len(out) // 2
    # If odd, the seed decides which side gets the extra item.
    if len(out) % 2 == 1 and int(seed) % 2 == 0:
        n_gold_a += 1

    positions = [True] * n_gold_a + [False] * (len(out) - n_gold_a)

    # Shuffle positions separately so gold A/B is balanced but not simply first half/second half.
    pos_keys = []
    for i in range(len(out)):
        digest = hashlib.md5(f"{seed}:position:{i}".encode()).hexdigest()
        pos_keys.append(int(digest[:12], 16))
    order = np.argsort(pos_keys)
    shuffled_positions = [None] * len(out)
    for pos_idx, original_idx in enumerate(order):
        shuffled_positions[original_idx] = positions[pos_idx]

    out["__gold_is_a"] = shuffled_positions
    return out.drop(columns=["__shuffle_key"])


def build_ab_pair(row: pd.Series, seed: int) -> Tuple[str, str, str, str]:
    physician = str(row["Physician Response"])
    chatgpt = str(row["ChatGPT Response"])
    gold_source = str(row["gold_source"])

    if gold_source == "Tie":
        raise ValueError("Cannot build A/B gold label for a tied average score row.")

    # Prefer the precomputed balanced assignment. Fall back to deterministic hash
    # only for safety if a row is inspected before assignment.
    if "__gold_is_a" in row.index and pd.notna(row.get("__gold_is_a")):
        gold_is_a = bool(row.get("__gold_is_a"))
    else:
        row_id = str(row["id"])
        gold_is_a = not deterministic_bool(seed, row_id)

    if gold_source == "ChatGPT":
        gold_answer = chatgpt
        other_answer = physician
        gold_name = "ChatGPT"
        other_name = "Physician"
    else:
        gold_answer = physician
        other_answer = chatgpt
        gold_name = "Physician"
        other_name = "ChatGPT"

    if gold_is_a:
        sentence_a = gold_answer
        sentence_b = other_answer
        order = f"{gold_name}_A__{other_name}_B"
        gold_choice = "A"
    else:
        sentence_a = other_answer
        sentence_b = gold_answer
        order = f"{other_name}_A__{gold_name}_B"
        gold_choice = "B"
    return sentence_a, sentence_b, order, gold_choice


def pred_source_from_order(response_order: str, pred_choice: str) -> str:
    """Map a predicted A/B letter back to Physician or ChatGPT."""
    pred_choice = str(pred_choice).strip().upper()
    response_order = str(response_order)
    if pred_choice not in {"A", "B"}:
        return ""
    parts = response_order.split("__")
    for part in parts:
        if part.endswith(f"_{pred_choice}"):
            return part.rsplit("_", 1)[0]
    return ""


def source_error_type(gold_source: str, pred_source: str) -> str:
    gold_source = str(gold_source).strip()
    pred_source = str(pred_source).strip()
    if pred_source not in {"Physician", "ChatGPT"}:
        return ""
    if gold_source == pred_source:
        return "Correct"
    if gold_source == "Physician" and pred_source == "ChatGPT":
        return "Judge favored ChatGPT; gold favored Physician"
    if gold_source == "ChatGPT" and pred_source == "Physician":
        return "Judge favored Physician; gold favored ChatGPT"
    return "Mismatch"

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
        question=str(row["Question"]),
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

    mismatch_explanation = ""
    mismatch_raw = ""

    if pred_choice != gold_choice:
        mismatch_text = safe_format_prompt(
            mismatch_prompt,
            question=str(row["Question"]),
            physician_answer=str(row["Physician Response"]),
            chatgpt_answer=str(row["ChatGPT Response"]),
            physician_avg=f"{float(row['physician_avg_score']):.3f}",
            chatgpt_avg=f"{float(row['chatgpt_avg_score']):.3f}",
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
        "pred_increased_empathy": int(pred_choice == gold_choice),
        "gold_increased_empathy": 1,
        "pred_choice": pred_choice,
        "gold_choice": gold_choice,
        "is_match": int(pred_choice == gold_choice),
        "gold_incremento_empatia_raw": str(row["gold_source"]),
        "empathy_reason": f"Judge chose {pred_choice}; gold choice is {gold_choice}. Gold source: {row['gold_source']}.",
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
        "gold_increased_empathy": 1,
        "pred_choice": np.nan,
        "gold_choice": gold_choice,
        "is_match": np.nan,
        "gold_incremento_empatia_raw": str(row.get("gold_source", "")),
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
        if "postID" in df.columns:
            df["id"] = df["postID"].astype(str)
        elif "Unnamed: 0" in df.columns:
            df["id"] = df["Unnamed: 0"].astype(str)
        else:
            df["id"] = df.index.astype(str)
    return df


def empty_results() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "id", "dataset_name", "question_text", "normal_answer_text", "empathy_answer_text",
        "judge_model", "api_model", "shot_mode", "score_dimension", "prompt_variant", "seed", "response_order",
        "gold_increased_empathy", "pred_increased_empathy", "pred_choice", "gold_choice", "is_match", "gold_incremento_empatia_raw",
        "empathy_reason", "mismatch_explanation", "exact_system_prompt", "exact_user_prompt",
        "raw_judge_response", "raw_mismatch_response", "judge_error", "judge_error_message", "scored_at",
        "physician_avg_score", "chatgpt_avg_score", "avg_score_diff_abs", "gold_source",
    ])



TEXT_RESULT_COLUMNS = [
    "id", "dataset_name", "question_text", "normal_answer_text", "empathy_answer_text",
    "judge_model", "api_model", "shot_mode", "score_dimension", "prompt_variant", "response_order", "pred_choice", "gold_choice",
    "gold_incremento_empatia_raw", "empathy_reason", "mismatch_explanation",
    "exact_system_prompt", "exact_user_prompt", "raw_judge_response", "raw_mismatch_response",
    "judge_error_message", "scored_at", "gold_source",
]

NUMERIC_RESULT_COLUMNS = [
    "seed", "gold_increased_empathy", "pred_increased_empathy", "is_match", "judge_error",
    "physician_avg_score", "chatgpt_avg_score", "avg_score_diff_abs",
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
    # Fresh-results version: load ONLY the selected Results CSV path.
    # Do not auto-merge legacy/current_run files, because that can show old
    # experiments in the dashboard and contaminate a fresh run.
    if not path or not os.path.exists(path):
        return empty_results()
    try:
        df = pd.read_csv(path)
    except Exception:
        return empty_results()
    df = ensure_results_schema(df)
    if "prompt_variant" in df.columns:
        df["prompt_variant"] = df["prompt_variant"].astype(str).replace({"nan": "", "None": ""})
    key_cols = [c for c in ["id", "judge_model", "seed", "score_dimension", "prompt_variant"] if c in df.columns]
    if key_cols:
        df = df.drop_duplicates(subset=key_cols, keep="last")
    return recover_choices_from_raw(df)


def save_results(df: pd.DataFrame, path: str) -> None:
    if path and os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    ensure_results_schema(df.copy()).to_csv(path, index=False)


def make_view(test_df: pd.DataFrame, results_df: pd.DataFrame, dataset_name: str, judge_model: str, seed: int, score_dimension: str, prompt_variant: str) -> pd.DataFrame:
    base_rows = []
    for _, r in test_df.iterrows():
        _, _, order, gold_choice = build_ab_pair(r, seed)
        base_rows.append({
            "id": str(r["id"]),
            "dataset_name": dataset_name,
            "question_text": str(r["Question"]),
            "normal_answer_text": str(r["Physician Response"]),
            "empathy_answer_text": str(r["ChatGPT Response"]),
            "gold_increased_empathy": 1,
            "score_dimension": score_dimension,
            "prompt_variant": prompt_variant,
            "gold_choice": gold_choice,
            "response_order": order,
            "gold_incremento_empatia_raw": str(r["gold_source"]),
            "broken_row_flag": False,
            "broken_row_reason": "",
            "physician_avg_score": float(r["physician_avg_score"]),
            "chatgpt_avg_score": float(r["chatgpt_avg_score"]),
            "avg_score_diff_abs": float(r["avg_score_diff_abs"]),
            "gold_source": str(r["gold_source"]),
        })
    base = pd.DataFrame(base_rows)
    results_df = ensure_results_schema(results_df.copy())
    subset = results_df[
        (results_df["judge_model"].astype(str) == judge_model)
        & (results_df["shot_mode"].astype(str) == "zero-shot")
        & (results_df["score_dimension"].astype(str) == str(score_dimension))
        & (
            (results_df["prompt_variant"].astype(str).fillna("") == str(prompt_variant))
            | ((str(prompt_variant) == "paper_minimal") & (results_df["prompt_variant"].astype(str).fillna("").isin(["", "nan", "None"])))
        )
        & (pd.to_numeric(results_df["seed"], errors="coerce").fillna(seed).astype(int) == int(seed))
    ].copy()
    subset["id"] = subset["id"].astype(str)
    merged = base.merge(
        subset.drop(columns=["dataset_name", "question_text", "normal_answer_text", "empathy_answer_text", "gold_increased_empathy", "gold_incremento_empatia_raw", "gold_choice", "response_order", "score_dimension", "prompt_variant", "physician_avg_score", "chatgpt_avg_score", "avg_score_diff_abs", "gold_source"], errors="ignore"),
        on="id",
        how="left",
    )
    merged["judge_model"] = merged.get("judge_model", judge_model).fillna(judge_model)
    merged["shot_mode"] = merged.get("shot_mode", "zero-shot").fillna("zero-shot")
    merged["seed"] = merged.get("seed", seed).fillna(seed).astype(int)
    merged["judge_error"] = pd.to_numeric(merged.get("judge_error", 0), errors="coerce").fillna(0).astype(int)
    return add_match_columns(recover_choices_from_raw(ensure_results_schema(merged)))

def upsert_view(all_results: pd.DataFrame, view_df: pd.DataFrame, judge_model: str, seed: int, score_dimension: str, prompt_variant: str) -> pd.DataFrame:
    all_results = ensure_results_schema(all_results.copy())
    mask = (
        (all_results["judge_model"].astype(str) == judge_model)
        & (all_results["shot_mode"].astype(str) == "zero-shot")
        & (all_results["score_dimension"].astype(str) == str(score_dimension))
        & (all_results["prompt_variant"].astype(str).fillna("") == str(prompt_variant))
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


def compute_source_metrics(view_df: pd.DataFrame) -> Dict[str, float]:
    done = view_df[view_df["pred_source"].astype(str).isin(["Physician", "ChatGPT"])].copy()
    done = done[done["gold_source"].astype(str).isin(["Physician", "ChatGPT"])].copy()
    if done.empty:
        return {}

    y_true = done["gold_source"].astype(str).values
    y_pred = done["pred_source"].astype(str).values
    labels = ["Physician", "ChatGPT"]
    match = y_true == y_pred

    return {
        "rows_scored": int(len(done)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "precision_physician": float(precision_score(y_true, y_pred, labels=["Physician"], average="macro", zero_division=0)),
        "recall_physician": float(recall_score(y_true, y_pred, labels=["Physician"], average="macro", zero_division=0)),
        "f1_physician": float(f1_score(y_true, y_pred, labels=["Physician"], average="macro", zero_division=0)),
        "precision_chatgpt": float(precision_score(y_true, y_pred, labels=["ChatGPT"], average="macro", zero_division=0)),
        "recall_chatgpt": float(recall_score(y_true, y_pred, labels=["ChatGPT"], average="macro", zero_division=0)),
        "f1_chatgpt": float(f1_score(y_true, y_pred, labels=["ChatGPT"], average="macro", zero_division=0)),
        "gold_physician": int((done["gold_source"].astype(str) == "Physician").sum()),
        "gold_chatgpt": int((done["gold_source"].astype(str) == "ChatGPT").sum()),
        "pred_physician": int((done["pred_source"].astype(str) == "Physician").sum()),
        "pred_chatgpt": int((done["pred_source"].astype(str) == "ChatGPT").sum()),
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


st.title("⚕️ JAMA Physician vs ChatGPT Pairwise Judge")
st.caption("Zero-shot only. The judge sees the physician response and ChatGPT response in balanced randomized A/B order and must answer only A or B. Gold labels are based on the higher average human evaluator empathy or quality score.")

with st.sidebar:
    st.header("Controls")

    default_judge_model = "+ llama-3.3-70b-versatile"
    default_judge_index = JUDGE_MODEL_LABELS.index(default_judge_model) if default_judge_model in JUDGE_MODEL_LABELS else 0
    model_choice = st.selectbox("Judge model label", JUDGE_MODEL_LABELS, index=default_judge_index)

    model_map = {
        "+ gemini-3.1-flash-lite": "gemini-3.1-flash-lite",
        "+ gpt-5.4-mini": "gpt-5.4-mini",
        "+ openai/gpt-oss-120b": "openai/gpt-oss-120b",
        "+ deepseek-v4-flash": "deepseek-v4-flash",
        "+ deepseek-v4-pro": "deepseek-v4-pro",
        "+ llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
        "+ llama-3.1-8b-instant": "llama-3.1-8b-instant",
        "+ venice-uncensored": "venice-uncensored",
    }

    def default_base_url_for_model(label: str) -> str:
        if label in {"+ deepseek-v4-flash", "+ deepseek-v4-pro"}:
            return "https://api.deepseek.com"
        if label == "+ gemini-3.1-flash-lite":
            return "https://generativelanguage.googleapis.com/v1beta/openai/"
        if label == "+ venice-uncensored":
            return "https://api.venice.ai/api/v1"
        if label == "+ gpt-5.4-mini":
            return ""
        return "https://api.groq.com/openai/v1"

    judge_model = model_choice
    api_model = model_map[model_choice]

    default_base_url = default_base_url_for_model(model_choice)
    # Streamlit keeps text_input state across reruns. Force the Base URL field to update
    # only when the selected model changes, while still allowing manual edits afterward.
    if st.session_state.get("_last_base_url_model_choice") != model_choice:
        st.session_state["base_url_input"] = default_base_url
        st.session_state["_last_base_url_model_choice"] = model_choice

    api_key = st.text_input("API key", type="password")
    base_url = st.text_input(
        "Base URL",
        key="base_url_input",
        help=(
            "Auto-filled by model: DeepSeek official = https://api.deepseek.com; "
            "Gemini OpenAI-compatible = https://generativelanguage.googleapis.com/v1beta/openai/; "
            "Venice = https://api.venice.ai/api/v1; OpenAI = blank; Groq = https://api.groq.com/openai/v1."
        ),
    )
    if model_choice in {"+ deepseek-v4-flash", "+ deepseek-v4-pro"}:
        st.info("DeepSeek official API selected. Use base URL https://api.deepseek.com and your DEEPSEEK API key.")
    elif model_choice == "+ gemini-3.1-flash-lite":
        st.info("Gemini OpenAI-compatible endpoint selected. Use base URL https://generativelanguage.googleapis.com/v1beta/openai/ and your Gemini API key.")
    elif model_choice == "+ venice-uncensored":
        st.info("Venice endpoint selected. Use base URL https://api.venice.ai/api/v1 and your Venice API key.")
    elif model_choice == "+ gpt-5.4-mini":
        st.info("OpenAI model selected. Leave Base URL blank and use your OpenAI API key.")

    st.text_input("Mode", value="zero-shot", disabled=True)
    score_dimension = st.selectbox("Human score dimension for gold label", ["Empathy", "Quality"], index=0)
    prompt_variant_label = st.selectbox("Prompt variant", [PROMPT_VARIANT_LABELS[v] for v in PROMPT_VARIANTS], index=0, help="Use paper_minimal as the current baseline, or paper_guided_editprompt as the alternate prompt for comparison.")
    prompt_variant = next(k for k, v in PROMPT_VARIANT_LABELS.items() if v == prompt_variant_label)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    batch_size = st.number_input("Number of rows to run now", min_value=1, value=50, step=1, help="This limits the Run next batch button. For example, 50 means score only the next 50 pending rows.")
    only_diff_ge_1 = st.checkbox("Only rows with average difference greater than or equal to 1", value=False, help="Results-view filter only. Runs always score all non-tie rows for the selected dimension; this checkbox only filters the displayed metrics/tables to rows where |average ChatGPT score - average physician score| >= 1.")
    with st.expander("Advanced: prompt order reproducibility", expanded=False):
        random_seed = st.number_input("Prompt order seed", min_value=0, value=42, step=1, help="Only controls whether the gold/preferred response is shown as A or B.")
    on_row_error = st.selectbox("If a row fails", ["Skip and continue", "Stop run after first failure"], index=1)
    test_file = st.file_uploader("Upload JAMA CSV", type=["csv"])
    results_path = st.text_input("Results CSV path", value=DEFAULT_RESULTS)

if not judge_model or not api_model:
    st.error("Choose or type a judge model.")
    st.stop()

try:
    test_df, test_name = load_jama_csv(test_file, DEFAULT_JAMA)
    test_df = ensure_id_column(test_df)
except Exception as e:
    st.error(f"Could not load JAMA CSV: {e}")
    st.stop()

required_cols = [
    "Question",
    "Physician Response",
    "ChatGPT Response",
    "Eval 1 Empathy (Physician)",
    "Eval 2 Empathy (Physician)",
    "Eval 3 Empathy (Physician)",
    "Eval 1 Empathy (ChatGPT)",
    "Eval 2 Empathy (ChatGPT)",
    "Eval 3 Empathy (ChatGPT)",
    "Eval 1 Quality (Physician)",
    "Eval 2 Quality (Physician)",
    "Eval 3 Quality (Physician)",
    "Eval 1 Quality (ChatGPT)",
    "Eval 2 Quality (ChatGPT)",
    "Eval 3 Quality (ChatGPT)",
]
missing = [c for c in required_cols if c not in test_df.columns]
if missing:
    st.error(f"JAMA CSV is missing required columns: {missing}")
    st.stop()

full_rows_count = len(test_df)
test_df = add_average_scores(test_df, score_dimension)
tie_rows_count = int((test_df["gold_source"] == "Tie").sum())
test_df = test_df[test_df["gold_source"] != "Tie"].copy().reset_index(drop=True)
non_tie_rows_count = len(test_df)

diff_ge_1_count = int((test_df["avg_score_diff_abs"] >= 1).sum())

# IMPORTANT:
# Balance A/B ONCE on the full non-tie set. The >=1 checkbox is a RESULTS-VIEW
# filter only: it must never change what rows are run, the A/B assignment,
# gold_choice, or saved predictions.
run_test_df = assign_balanced_ab_order(test_df, int(random_seed)).reset_index(drop=True)

all_results = load_results(results_path)
run_view_df = coerce_result_dtypes(add_match_columns(make_view(run_test_df, all_results, test_name, judge_model, int(random_seed), score_dimension, prompt_variant)))

# Map A/B letters back to the real response source on the full run dataframe.
run_view_df["pred_source"] = run_view_df.apply(lambda r: pred_source_from_order(r.get("response_order", ""), r.get("pred_choice", "")), axis=1)
run_view_df["source_match"] = (run_view_df["gold_source"].astype(str) == run_view_df["pred_source"].astype(str)) & run_view_df["pred_source"].isin(["Physician", "ChatGPT"])
run_view_df["source_error_type"] = run_view_df.apply(lambda r: source_error_type(r.get("gold_source", ""), r.get("pred_source", "")), axis=1)

# Display/results dataframe. This is the only place where the checkbox applies.
if only_diff_ge_1:
    view_df = run_view_df[run_view_df["avg_score_diff_abs"] >= 1].copy().reset_index(drop=True)
else:
    view_df = run_view_df.copy().reset_index(drop=True)

st.subheader("Current configuration")
c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Judge label", judge_model)
c2.metric("API model", api_model)
c3.metric("Mode", "zero-shot")
c4.metric("Rows used", len(test_df))
c5.metric("Tied rows excluded", tie_rows_count)
c6.metric("Prompt order seed", int(random_seed))
c7.metric("Prompt variant", prompt_variant)
st.caption("Stable A/B note: A/B prompt order is assigned once on the full non-tie set. The >=1 checkbox is only a results-view filter and never changes the run queue or saved predictions.")
st.caption(f"Full CSV rows: {full_rows_count} · Run rows/non-tie rows: {non_tie_rows_count} · Displayed rows: {len(view_df)} · Tied rows excluded: {tie_rows_count} · Rows with avg difference >= 1: {diff_ge_1_count} · Difference filter enabled for results view: {only_diff_ge_1} · Score dimension: {score_dimension} · Prompt variant: {prompt_variant}")

st.subheader("Prompt templates")
default_system_prompt, default_user_prompt, default_mismatch_prompt = get_default_prompts(score_dimension, prompt_variant)

# When switching Empathy ↔ Quality, update the prompt text automatically so the
# actual model instruction matches the selected gold-score dimension.
if "prompt_score_dimension" not in st.session_state:
    st.session_state["prompt_score_dimension"] = score_dimension
if "prompt_variant" not in st.session_state:
    st.session_state["prompt_variant"] = prompt_variant
if "system_prompt" not in st.session_state:
    st.session_state["system_prompt"] = default_system_prompt
if "user_prompt" not in st.session_state:
    st.session_state["user_prompt"] = default_user_prompt
if "mismatch_prompt" not in st.session_state:
    st.session_state["mismatch_prompt"] = default_mismatch_prompt

if st.session_state.get("prompt_score_dimension") != score_dimension or st.session_state.get("prompt_variant") != prompt_variant:
    st.session_state["system_prompt"] = default_system_prompt
    st.session_state["user_prompt"] = default_user_prompt
    st.session_state["mismatch_prompt"] = default_mismatch_prompt
    st.session_state["prompt_score_dimension"] = score_dimension
    st.session_state["prompt_variant"] = prompt_variant

if st.button(f"Reset prompts to {score_dimension} / {prompt_variant} defaults"):
    st.session_state["system_prompt"] = default_system_prompt
    st.session_state["user_prompt"] = default_user_prompt
    st.session_state["mismatch_prompt"] = default_mismatch_prompt
    st.session_state["prompt_score_dimension"] = score_dimension
    st.session_state["prompt_variant"] = prompt_variant

st.text_area("System prompt", key="system_prompt", height=130)
st.text_area("Pairwise evaluation prompt", key="user_prompt", height=430)
st.text_area("Mismatch explanation prompt", key="mismatch_prompt", height=230)

completed_mask = view_df["pred_choice"].astype(str).isin(["A", "B"])
failed_mask = view_df["judge_error"].fillna(0).astype(int).eq(1)

metrics = compute_metrics(view_df)
source_metrics = compute_source_metrics(view_df)

## Metrics and matrices are displayed below the run controls.

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
selected_ids = st.multiselect("Specific IDs to run", options=run_view_df["id"].astype(str).tolist(), default=[])
run_selected = b1.button("▶ Run selected IDs", type="primary", use_container_width=True)
run_next = b2.button(f"▶ Run next {int(batch_size)} pending rows", use_container_width=True)
run_all = b3.button("▶ Run all remaining", use_container_width=True)
reset_current = b4.button("🗑 Reset current model/seed", use_container_width=True)

if reset_current:
    run_view_df = coerce_result_dtypes(add_match_columns(make_view(run_test_df, empty_results(), test_name, judge_model, int(random_seed), score_dimension, prompt_variant)))
    all_results = upsert_view(all_results, run_view_df, judge_model, int(random_seed), score_dimension, prompt_variant)
    save_results(all_results, results_path)
    st.success("Current model / prompt-order-seed results reset.")
    st.rerun()

progress_slot = st.empty()
status_slot = st.empty()


def run_rows(target_indices: List[int], stop_on_error: bool) -> Dict[str, Any]:
    global run_view_df
    run_view_df = coerce_result_dtypes(run_view_df)
    progress = progress_slot.progress(0)
    attempted = 0
    failed = 0
    stopped_early = False
    last_failed_id = None
    last_failed_error = ""

    for n, idx in enumerate(target_indices, start=1):
        row = run_test_df.loc[run_test_df["id"].astype(str) == str(run_view_df.loc[idx, "id"])].iloc[0]
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
            if key not in run_view_df.columns:
                run_view_df[key] = pd.Series([""] * len(run_view_df), dtype="object")
            if key in TEXT_RESULT_COLUMNS or isinstance(value, str):
                run_view_df[key] = run_view_df[key].astype("object")
            run_view_df.at[idx, key] = value

        run_view_df.at[idx, "judge_model"] = judge_model
        run_view_df.at[idx, "api_model"] = api_model
        run_view_df.at[idx, "shot_mode"] = "zero-shot"
        run_view_df.at[idx, "score_dimension"] = score_dimension
        run_view_df.at[idx, "prompt_variant"] = prompt_variant
        run_view_df.at[idx, "seed"] = int(random_seed)
        run_view_df.at[idx, "scored_at"] = pd.Timestamp.utcnow().isoformat()

        if int(pred.get("judge_error", 0)) == 1:
            failed += 1
            last_failed_id = row["id"]
            last_failed_error = str(pred.get("judge_error_message", ""))
            if stop_on_error:
                stopped_early = True
                break

        all_current = upsert_view(load_results(results_path), run_view_df, judge_model, int(random_seed), score_dimension, prompt_variant)
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
        pending_indices = run_view_df.index[~run_view_df["pred_choice"].astype(str).isin(["A", "B"])].tolist()
        if run_selected:
            target_indices = run_view_df.index[run_view_df["id"].astype(str).isin([str(x) for x in selected_ids])].tolist()
        elif run_next:
            target_indices = pending_indices[:int(batch_size)]
        else:
            target_indices = pending_indices

        if not target_indices:
            st.warning("No rows selected to run.")
        else:
            st.session_state["last_run_summary"] = run_rows(target_indices, stop_on_error=(on_row_error == "Stop run after first failure"))
            st.rerun()


st.subheader("Source preference matrix")
st.caption("Main interpretation matrix. It maps the judge's A/B answer back to Physician vs ChatGPT.")

source_done = view_df[view_df["pred_source"].astype(str).isin(["Physician", "ChatGPT"])].copy()
if source_done.empty:
    st.info("No scored rows yet for source-preference matrix.")
else:
    source_cm_df = pd.crosstab(
        source_done["gold_source"].astype(str),
        source_done["pred_source"].astype(str),
        rownames=["Gold source from human avg"],
        colnames=["Judge predicted source"],
        dropna=False,
    ).reindex(index=["Physician", "ChatGPT"], columns=["Physician", "ChatGPT"], fill_value=0)

    # Requested visual orientation:
    # top row = ChatGPT, bottom row = Physician;
    # left column = Physician, right column = ChatGPT.
    source_cm_display_df = source_cm_df.reindex(index=["ChatGPT", "Physician"], columns=["Physician", "ChatGPT"], fill_value=0)
    st.dataframe(source_cm_display_df, use_container_width=True)

    fig_source = px.imshow(
        source_cm_display_df,
        text_auto=True,
        aspect="auto",
        origin="upper",
        labels=dict(x="Judge predicted source", y="Gold source from human avg", color="Count"),
        title="Source preference matrix",
    )
    fig_source.update_yaxes(categoryorder="array", categoryarray=["ChatGPT", "Physician"])
    fig_source.update_xaxes(categoryorder="array", categoryarray=["Physician", "ChatGPT"])
    st.plotly_chart(fig_source, use_container_width=True, key="source_preference_matrix")

    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Gold Physician / Pred Physician", int(source_cm_df.loc["Physician", "Physician"]))
    sc2.metric("Gold Physician / Pred ChatGPT", int(source_cm_df.loc["Physician", "ChatGPT"]))
    sc3.metric("Gold ChatGPT / Pred Physician", int(source_cm_df.loc["ChatGPT", "Physician"]))
    sc4.metric("Gold ChatGPT / Pred ChatGPT", int(source_cm_df.loc["ChatGPT", "ChatGPT"]))

    mismatch_by_source = source_done[source_done["gold_source"].astype(str) != source_done["pred_source"].astype(str)].copy()
    if not mismatch_by_source.empty:
        st.markdown("**Mismatch direction counts**")
        st.dataframe(mismatch_by_source["source_error_type"].value_counts().rename_axis("Mismatch direction").reset_index(name="count"), use_container_width=True)

if source_metrics:
    st.subheader("Source-based metrics")
    st.caption("Primary JAMA metrics. These treat Physician and ChatGPT as the two classes, independent of randomized A/B position.")
    sm1, sm2, sm3, sm4, sm5, sm6 = st.columns(6)
    sm1.metric("Rows scored", source_metrics.get("rows_scored", 0))
    sm2.metric("Source accuracy", f"{source_metrics['accuracy']:.3f}")
    sm3.metric("Source macro F1", f"{source_metrics['macro_f1']:.3f}")
    sm4.metric("Source macro precision", f"{source_metrics['macro_precision']:.3f}")
    sm5.metric("Source macro recall", f"{source_metrics['macro_recall']:.3f}")
    sm6.metric("Source mismatches", source_metrics.get("mismatches", 0))

    sm7, sm8, sm9, sm10 = st.columns(4)
    sm7.metric("Gold Physician", source_metrics.get("gold_physician", 0))
    sm8.metric("Gold ChatGPT", source_metrics.get("gold_chatgpt", 0))
    sm9.metric("Pred Physician", source_metrics.get("pred_physician", 0))
    sm10.metric("Pred ChatGPT", source_metrics.get("pred_chatgpt", 0))

    with st.expander("Detailed source precision / recall / F1", expanded=True):
        source_metric_table = pd.DataFrame([
            {"label": "Physician", "precision": source_metrics["precision_physician"], "recall": source_metrics["recall_physician"], "f1": source_metrics["f1_physician"]},
            {"label": "ChatGPT", "precision": source_metrics["precision_chatgpt"], "recall": source_metrics["recall_chatgpt"], "f1": source_metrics["f1_chatgpt"]},
            {"label": "macro avg", "precision": source_metrics["macro_precision"], "recall": source_metrics["macro_recall"], "f1": source_metrics["macro_f1"]},
            {"label": "weighted avg", "precision": source_metrics["weighted_precision"], "recall": source_metrics["weighted_recall"], "f1": source_metrics["weighted_f1"]},
        ])
        st.dataframe(source_metric_table, use_container_width=True, hide_index=True)
        st.caption("Precision: when the judge selects a source, how often is it the human-average gold source? Recall: among rows where a source is the human-average gold source, how often does the judge select it? Physician metrics can look low when only a few rows have Physician as gold.")

st.subheader("A/B metrics")
st.caption("Diagnostic only. A and B are randomized positions, not the real response sources.")
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Rows scored", metrics.get("rows_scored", 0))
m2.metric("A/B accuracy", f"{metrics['accuracy']:.3f}" if metrics else "")
m3.metric("A/B macro F1", f"{metrics['macro_f1']:.3f}" if metrics else "")
m4.metric("A/B macro precision", f"{metrics['macro_precision']:.3f}" if metrics else "")
m5.metric("A/B macro recall", f"{metrics['macro_recall']:.3f}" if metrics else "")
m6.metric("A/B mismatches", metrics.get("mismatches", 0) if metrics else 0)

m7, m8, m9, m10, m11, m12 = st.columns(6)
m7.metric("Gold A rows", int((view_df["gold_choice"].astype(str) == "A").sum()))
m8.metric("Gold B rows", int((view_df["gold_choice"].astype(str) == "B").sum()))
m9.metric("Pred A", int((view_df["pred_choice"].astype(str) == "A").sum()))
m10.metric("Pred B", int((view_df["pred_choice"].astype(str) == "B").sum()))
m11.metric("Failed rows", int(failed_mask.sum()))
m12.metric("Prompt order split", f"Gold=A {int((view_df['gold_choice'].astype(str) == 'A').sum())} / Gold=B {int((view_df['gold_choice'].astype(str) == 'B').sum())}")

if metrics:
    with st.expander("Detailed A/B precision / recall / F1", expanded=False):
        metric_table = pd.DataFrame([
            {"label": "A", "precision": metrics["precision_A"], "recall": metrics["recall_A"], "f1": metrics["f1_A"]},
            {"label": "B", "precision": metrics["precision_B"], "recall": metrics["recall_B"], "f1": metrics["f1_B"]},
            {"label": "macro avg", "precision": metrics["macro_precision"], "recall": metrics["macro_recall"], "f1": metrics["macro_f1"]},
            {"label": "weighted avg", "precision": metrics["weighted_precision"], "recall": metrics["weighted_recall"], "f1": metrics["weighted_f1"]},
        ])
        st.dataframe(metric_table, use_container_width=True, hide_index=True)

st.subheader("A/B matrix view")
st.caption("Position/randomization diagnostic. It checks whether the judge behaves differently when the gold response is shown as A or B.")

done_for_cm = view_df[view_df["pred_choice"].astype(str).isin(["A", "B"])].copy()
if done_for_cm.empty:
    gold_counts = view_df["gold_choice"].astype(str).value_counts().reindex(["A", "B"], fill_value=0)
    st.info("No scored rows yet for this selected model / prompt-order-seed / score dimension / prompt variant / results file.")
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

    choice_cm_display_df = choice_cm_df.reindex(index=["B", "A"], columns=["A", "B"], fill_value=0)
    st.dataframe(choice_cm_display_df, use_container_width=True)
    fig_choice = px.imshow(choice_cm_display_df, text_auto=True, aspect="auto", origin="upper", labels=dict(x="Predicted letter", y="Gold letter", color="Count"), title="A/B choice matrix")
    fig_choice.update_yaxes(categoryorder="array", categoryarray=["B", "A"])
    fig_choice.update_xaxes(categoryorder="array", categoryarray=["A", "B"])
    st.plotly_chart(fig_choice, use_container_width=True, key="choice_matrix_chart_bottom")

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
preview_mode = st.radio("Show rows", ["All rows", "Pending only", "Scored only", "Failed only", "Correct only", "Mismatches only", "Gold A / Pred A", "Gold A / Pred B", "Gold B / Pred A", "Gold B / Pred B", "Gold Physician / Pred ChatGPT", "Gold ChatGPT / Pred Physician"], horizontal=True, index=0)
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
elif preview_mode in ["Gold Physician / Pred ChatGPT", "Gold ChatGPT / Pred Physician"]:
    parts = preview_mode.replace("Gold ", "").replace("Pred ", "").split(" / ")
    gold_source_filter, pred_source_filter = parts[0], parts[1]
    preview_df = view_df[(view_df["gold_source"].astype(str) == gold_source_filter) & (view_df["pred_source"].astype(str) == pred_source_filter)].copy()
elif preview_mode.startswith("Gold"):
    gold_letter = preview_mode.split()[1]
    pred_letter = preview_mode.split()[-1]
    preview_df = view_df[(view_df["gold_choice"].astype(str) == gold_letter) & (view_df["pred_choice"].astype(str) == pred_letter)].copy()
else:
    preview_df = view_df.copy()

show_cols = [
    "id", "gold_choice", "pred_choice", "is_match", "gold_increased_empathy", "pred_increased_empathy", "gold_incremento_empatia_raw", "response_order",
    "empathy_reason", "mismatch_explanation", "judge_error", "judge_error_message",
    "question_text", "normal_answer_text", "empathy_answer_text", "physician_avg_score", "chatgpt_avg_score", "avg_score_diff_abs", "gold_source", "pred_source", "source_error_type", "exact_user_prompt",
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
elif inspect_filter in ["Gold Physician / Pred ChatGPT", "Gold ChatGPT / Pred Physician"]:
    parts = inspect_filter.replace("Gold ", "").replace("Pred ", "").split(" / ")
    gold_source_filter, pred_source_filter = parts[0], parts[1]
    inspect_source_df = inspect_source_df[(inspect_source_df["gold_source"].astype(str) == gold_source_filter) & (inspect_source_df["pred_source"].astype(str) == pred_source_filter)]
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
        st.markdown("**Patient question**")
        st.write(row["question_text"])
        st.markdown("**Physician response**")
        st.write(row["normal_answer_text"])
    with c2:
        st.markdown("**ChatGPT response**")
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
                <div><strong>Gold source:</strong> {row.get('gold_source', '')}</div>
                <div><strong>Pred source:</strong> {row.get('pred_source', '')}</div>
                <div><strong>Error type:</strong> {row.get('source_error_type', '')}</div>
                <div><strong>Physician avg:</strong> {row.get('physician_avg_score', '')}</div>
                <div><strong>ChatGPT avg:</strong> {row.get('chatgpt_avg_score', '')}</div>
                <div><strong>Avg diff:</strong> {row.get('avg_score_diff_abs', '')}</div>
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

current_run_download_df = ensure_results_schema(run_view_df.copy())
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
