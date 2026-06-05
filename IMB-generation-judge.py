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

st.set_page_config(page_title="IMB Empathy Generation + Judge", page_icon="🇮🇹", layout="wide")

DEFAULT_DATASET = "IMB_QA_matched_subset_1000.csv"
DATASET_VARIANTS = {
    "Normal answers (40–180 words)": {
        "key": "normal",
        "path": "IMB_QA_matched_subset_1000.csv",
    },
    "Long answers (180–350 words)": {
        "key": "long",
        "path": "IMB_QA_long_answers_subset_1000.csv",
    },
}

GENERATION_PROMPT_VARIANTS = {
    "V1: IDRE/IMB empathy rewrite (original prompt)": {
        "key": "v1",
        "kind": "idre_imb",
        "length_rule": "",
    },
    "V2: IDRE/IMB empathy rewrite + similar length": {
        "key": "v2_length_matched",
        "kind": "idre_imb",
        "length_rule": "- Keep the rewritten answer similar in length to the original answer.\n",
    },
    "V3: Refined clinical empathy editing prompt": {
        "key": "v3_refined_clinical_editing",
        "kind": "refined_clinical_editing",
        "length_rule": "",
    },
    "V4: Concise clinical empathy editing prompt": {
        "key": "v4_concise_clinical_editing",
        "kind": "concise_clinical_editing",
        "length_rule": "",
    },
}

LEGACY_LLAMA_GENERATION_RESULTS = "current_run_IMB_GENERATED_BY_LLAMA_seed-42.csv"
LEGACY_LLAMA_GPTOSS_JUDGE_RESULTS = "current_run_IMB_JUDGED_BY_plus_openai_gpt_oss_120b_seed-42.csv"
LEGACY_SINGLE_RESULTS = "current_run_IMB_empathy_quality_JAMA_PROMPTS_EXACT_seed-42.csv"

GENERATION_MODEL_LABELS = [
    "+ llama-3.3-70b-versatile",
    "+ openai/gpt-oss-120b",
]

JUDGE_MODEL_LABELS = [
    "+ llama-3.3-70b-versatile",
    "+ openai/gpt-oss-120b",
]

MODEL_TO_API_MODEL = {
    "+ llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
    "+ llama-3.1-8b-instant": "llama-3.1-8b-instant",
    "+ openai/gpt-oss-120b": "openai/gpt-oss-120b",
    "+ deepseek-v4-flash": "deepseek-v4-flash",
    "+ deepseek-v4-pro": "deepseek-v4-pro",
    "+ venice-uncensored": "dolphin-2.9.2-qwen2-72b",
    "+ gemini-3.1-flash-lite": "gemini-3.1-flash-lite",
    "+ gpt-5.4-mini": "gpt-5.4-mini",
}

MODEL_DEFAULT_BASE_URL = {
    "+ llama-3.3-70b-versatile": "https://api.groq.com/openai/v1",
    "+ llama-3.1-8b-instant": "https://api.groq.com/openai/v1",
    "+ openai/gpt-oss-120b": "https://api.groq.com/openai/v1",
    "+ deepseek-v4-flash": "https://api.deepseek.com",
    "+ deepseek-v4-pro": "https://api.deepseek.com",
    "+ venice-uncensored": "https://api.venice.ai/api/v1",
    "+ gemini-3.1-flash-lite": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "+ gpt-5.4-mini": "",
}

TEXT_COLUMNS = [
    "id", "general_category", "category", "date", "url", "question", "original_answer",
    "generated_empathy_answer", "generation_model", "generation_prompt_variant", "generated_at", "generation_error",
    "response_order", "sentence_a_source", "sentence_b_source", "gold_choice",
    "empathy_pred_choice", "empathy_pred_source", "empathy_raw_judge_response",
    "empathy_judge_model", "empathy_judged_at", "empathy_judge_error",
    "quality_pred_choice", "quality_pred_source", "quality_raw_judge_response",
    "quality_judge_model", "quality_judged_at", "quality_judge_error",
    "empathy_mismatch_explanation", "empathy_mismatch_raw_response", "empathy_mismatch_explained_at", "empathy_mismatch_error",
    "quality_mismatch_explanation", "quality_mismatch_raw_response", "quality_mismatch_explained_at", "quality_mismatch_error",
    "factual_preservation_answer", "factual_preservation_raw_response", "factual_preservation_judged_at", "factual_preservation_error",
    "unsupported_addition_answer", "unsupported_addition_raw_response", "unsupported_addition_judged_at", "unsupported_addition_error",
    "factual_preservation_judge_model", "unsupported_addition_judge_model",
    # Legacy/scratch columns kept for compatibility if an old CSV is accidentally loaded.
    "pred_choice", "pred_source", "raw_judge_response", "judge_model", "judged_at", "judge_error",
]

NUMERIC_COLUMNS = ["question_word_count", "answer_word_count"]


def normalize_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).replace("\r\n", "\n").replace("\r", "\n").strip()


def stable_hash_int(*parts: Any) -> int:
    s = "||".join(str(p) for p in parts)
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def slugify_label(label: str) -> str:
    s = normalize_text(label).lower().replace("+", "plus")
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "model"


def variant_suffix(prompt_variant_key: str) -> str:
    # V1 keeps old filenames so previously generated/judged results remain visible.
    key = normalize_text(prompt_variant_key) or "v1"
    return "" if key == "v1" else f"_{key.upper()}"


def normalize_prompt_variant_value(value: Any) -> str:
    """Treat legacy blank prompt-variant values as V1.

    Old CSVs were created before generation_prompt_variant existed, so their
    variant column is blank. Those rows are V1, not invalid.
    """
    v = normalize_text(value)
    if not v or v.lower() in {"nan", "none"}:
        return "v1"
    return v




def infer_prompt_variant_from_filename(path: str) -> str:
    """Infer generation prompt variant from legacy/current filename.

    This is intentionally conservative. Older CSVs often have a blank
    generation_prompt_variant column, but the filename clearly contains V3/V4.
    Without this, the dashboard treats them as V1 and hides generated outputs.
    """
    name = os.path.basename(str(path)).upper().replace(" ", "_")
    if "V4_CONCISE_CLINICAL_EDITING" in name:
        return "v4_concise_clinical_editing"
    if "V3_REFINED_CLINICAL_EDITING" in name:
        return "v3_refined_clinical_editing"
    if "V2_LENGTH_MATCHED" in name:
        return "v2_length_matched"
    return "v1"


def repair_prompt_variant_from_filename(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """Backfill prompt variant using filename when CSV metadata is blank/legacy.

    Only changes blank/nan/none/v1 rows when the filename says V2/V3/V4.
    """
    inferred = infer_prompt_variant_from_filename(path)
    if "generation_prompt_variant" not in df.columns:
        df["generation_prompt_variant"] = ""
    vals = df["generation_prompt_variant"].apply(normalize_prompt_variant_value)
    if inferred != "v1":
        mask = vals.isin(["", "v1", "nan", "none"])
        df.loc[mask, "generation_prompt_variant"] = inferred
    else:
        df["generation_prompt_variant"] = vals
    return df

def generation_results_path_for(gen_label: str, seed: int, subset_key: str = "normal", prompt_variant_key: str = "v1") -> str:
    # Keep already-created V1 filenames for compatibility.
    subset_key = normalize_text(subset_key) or "normal"
    suffix = "" if subset_key == "normal" else f"_{subset_key.upper()}"
    pv_suffix = variant_suffix(prompt_variant_key)
    if gen_label == "+ llama-3.3-70b-versatile":
        return f"current_run_IMB{suffix}{pv_suffix}_GENERATED_BY_LLAMA_seed-{int(seed)}.csv"
    return f"current_run_IMB{suffix}{pv_suffix}_GENERATED_BY_{slugify_label(gen_label)}_seed-{int(seed)}.csv"


def judge_results_path_for(gen_label: str, judge_label: str, seed: int, subset_key: str = "normal", prompt_variant_key: str = "v1") -> str:
    # Judge files depend on subset + generation prompt variant + generator + judge.
    # V1 uses old filenames so previous results remain visible.
    subset_key = normalize_text(subset_key) or "normal"
    suffix = "" if subset_key == "normal" else f"_{subset_key.upper()}"
    pv_suffix = variant_suffix(prompt_variant_key)
    gen_slug = "llama" if gen_label == "+ llama-3.3-70b-versatile" else slugify_label(gen_label)
    return f"current_run_IMB{suffix}{pv_suffix}_GENERATED_BY_{gen_slug}_JUDGED_BY_{slugify_label(judge_label)}_seed-{int(seed)}.csv"


def api_model_name(label: str) -> str:
    return MODEL_TO_API_MODEL.get(label, label.replace("+ ", ""))


def make_client(api_key: str, base_url: str) -> OpenAI:
    kwargs = {"api_key": api_key}
    if normalize_text(base_url):
        kwargs["base_url"] = normalize_text(base_url)
    return OpenAI(**kwargs)


def ask_model(client: OpenAI, model: str, system_prompt: str, user_prompt: str, temperature: float = 0.0, max_tokens: int | None = None) -> str:
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    # max_tokens is intentionally optional. If it is None, no explicit output cap is sent.
    # This avoids imposing an artificial experimental limit on generation or judging.
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


def humanize_api_error(e: Exception) -> str:
    """Return a clear dashboard error for API/quota/rate-limit failures."""
    raw = str(e)
    low = raw.lower()
    status = getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)
    err_type = e.__class__.__name__

    if status in {401, 403} or "invalid api key" in low or "authentication" in low or "unauthorized" in low:
        reason = "Authentication/API key error"
    elif status == 402 or "insufficient" in low or "balance" in low or "quota" in low or "billing" in low or "credits" in low:
        reason = "Quota/balance exhausted"
    elif status == 429 or "rate limit" in low or "too many requests" in low:
        reason = "Rate limit"
    elif status and status >= 500:
        reason = "Provider/server error"
    else:
        reason = "API call failed"

    shown = raw.replace("\n", " ").strip()
    if len(shown) > 900:
        shown = shown[:900] + "..."
    return f"{reason} ({err_type}{', HTTP ' + str(status) if status else ''}): {shown}"


def generation_prompts(prompt_variant_key: str = "v1") -> Tuple[str, str]:
    variant = None
    for item in GENERATION_PROMPT_VARIANTS.values():
        if item["key"] == prompt_variant_key:
            variant = item
            break
    if variant is None:
        variant = GENERATION_PROMPT_VARIANTS["V1: IDRE/IMB empathy rewrite (original prompt)"]

    if variant.get("kind") == "concise_clinical_editing":
        system_prompt = "You are a careful Italian medical communication editor. Follow the user instruction exactly."
        user_template = """Rewrite the Italian medical answer to make it warmer, more empathetic, and more concise than the original.

Key editing principles:

1. Preserve factual and clinical accuracy.
- Do not introduce or infer medical facts or conditions not explicitly stated in the original response.
- Do not make any clinical assumptions or diagnoses that are not present in the original response.

2. Respect the original response's intent.
- Do not add follow-up recommendations or next-step suggestions unless they already appear in the original response.
- Do not add new advice, warnings, or treatment instructions.

3. Maintain emotional balance.
- Do not add false reassurance or overconfidence.
- Do not introduce unnecessary doubt, fear, or alarming language.
- Empathy should be expressed through tone, acknowledgment, and understanding, not through added medical content.

4. Make the response concise.
- Make the rewritten answer shorter than the original when possible.
- Remove redundancy, repetition, and unnecessary wording.
- Preserve all clinically important information.
- Maintain a professional medical tone.

5. Output requirements.
- Keep the answer in Italian.
- Write only the rewritten answer, without explanations or labels.

Patient question:
{question}

Original medical answer:
{original_answer}

Concise empathy-edited answer:""".strip()
        return system_prompt, user_template

    if variant.get("kind") == "refined_clinical_editing":
        system_prompt = "You are a careful Italian medical communication editor. Follow the user instruction exactly."
        user_template = """Rewrite the Italian medical answer to make it more empathetic and supportive.

Key editing principles:

1. Preserve factual and clinical accuracy.
- Do not introduce or infer medical facts or conditions not explicitly stated in the original response.
- Do not make any clinical assumptions or diagnoses that are not present in the original response.

2. Respect the original response's intent.
- Do not add follow-up recommendations or next-step suggestions unless they already appear in the original response.
- Do not add new advice, warnings, or treatment instructions.

3. Maintain emotional balance.
- Do not add false reassurance or overconfidence.
- Do not introduce unnecessary doubt, fear, or alarming language.
- Empathy should be expressed through tone, acknowledgment, and understanding, not through added medical content.

4. Preserve structure and style.
- Keep the response roughly the same length as the original.
- Maintain the professional medical tone and sentence structure where possible.
- Revise only what's necessary to make the tone warmer, more understanding, or more supportive.

5. Output requirements.
- Keep the answer in Italian.
- Write only the rewritten answer, without explanations or labels.

Patient question:
{question}

Original medical answer:
{original_answer}

Rewritten answer:""".strip()
        return system_prompt, user_template

    system_prompt = "You are a careful Italian medical communication editor. Follow the user instruction exactly."
    length_rule = variant.get("length_rule", "")
    user_template = f"""Rewrite the Italian medical answer to make it more empathetic and supportive.

Rules:
- Keep the same medical meaning.
- Do not add new medical facts.
- Do not remove important medical information.
- Do not give a diagnosis beyond the original answer.
- Do not mention that you are an AI.
- Keep the answer in Italian.
{length_rule}- Write only the rewritten answer, without explanations or labels.

Patient question:
{{question}}

Original medical answer:
{{original_answer}}

Empathy-augmented answer:""".strip()
    return system_prompt, user_template


def judge_prompts(dimension: str) -> Tuple[str, str]:
    """Return the exact JAMA-style prompts used in the previous experiment.

    The same generic system prompt is used for both dimensions.
    Only the user prompt changes between empathy and quality.
    """
    dim = normalize_text(dimension).lower()
    system_prompt = "You are an AI judge. Follow the user instruction exactly and answer with only A or B."

    if dim == "quality":
        user_template = """Given a patient question and two responses, which response has higher quality?
Answer with only A or B.

Patient question:
{question}

A. {sentence_a}

B. {sentence_b}""".strip()
    else:
        user_template = """Given a patient question and two responses, which response has higher empathy?
Answer with only A or B.

Patient question:
{question}

A. {sentence_a}

B. {sentence_b}""".strip()
    return system_prompt, user_template




def factual_check_prompts(check_type: str) -> Tuple[str, str]:
    """Fact-preservation checks inspired by clinical empathy-editing evaluation.

    These are not broad quality judgments. They test whether the edited answer
    preserves original clinical information and avoids unsupported medical additions.
    """
    ct = normalize_text(check_type).lower()
    system_prompt = "You are an AI judge. Follow the user instruction exactly and answer with only Yes or No."
    if ct == "unsupported_addition":
        user_template = """Given a patient question, an original medical answer, and a rewritten answer, does the rewritten answer introduce new medical facts, conditions, advice, warnings, diagnoses, treatment instructions, or follow-up recommendations that were not present in the original answer?
Answer with only Yes or No.

Patient question:
{question}

Original medical answer:
{original_answer}

Rewritten answer:
{generated_answer}""".strip()
    else:
        user_template = """Given a patient question, an original medical answer, and a rewritten answer, does the rewritten answer preserve the clinically important information from the original answer?
Answer with only Yes or No.

Patient question:
{question}

Original medical answer:
{original_answer}

Rewritten answer:
{generated_answer}""".strip()
    return system_prompt, user_template


def parse_yes_no(raw: str) -> str:
    text = normalize_text(raw).lower()
    if text in {"yes", "yes."} or text.startswith("yes\n") or text.startswith("yes "):
        return "Yes"
    if text in {"no", "no."} or text.startswith("no\n") or text.startswith("no "):
        return "No"
    m = re.search(r"\b(yes|no)\b", text, flags=re.I)
    return m.group(1).capitalize() if m else ""


def exact_factual_user_prompt(row: pd.Series, check_type: str) -> str:
    _sys, user_template = factual_check_prompts(check_type)
    return user_template.format(
        question=normalize_text(row.get("question", "")),
        original_answer=normalize_text(row.get("original_answer", "")),
        generated_answer=normalize_text(row.get("generated_empathy_answer", "")),
    )



def mismatch_explanation_prompts(dimension: str) -> Tuple[str, str]:
    """Prompt used only for explaining cases where the generated answer was not preferred.

    This is not part of the main preference judgment. It is a post-hoc diagnostic step.
    """
    dim = normalize_text(dimension).lower()
    system_prompt = "You are an AI judge. Explain the model preference briefly and concretely."
    if dim == "quality":
        user_template = """The judge preferred the original response over the empathy-augmented response for QUALITY.
Explain why the generated empathy-augmented response may have been judged lower in quality.
Focus on concrete differences such as medical accuracy, completeness, clarity, relevance, unnecessary additions, or loss of important information.
Do not invent facts. Use only the texts below.
Write 1-3 concise sentences.

Patient question:
{question}

Original response:
{original_answer}

Empathy-augmented response:
{generated_answer}

Judge raw answer:
{raw_judge_response}

Explanation:""".strip()
    else:
        user_template = """The judge preferred the original response over the empathy-augmented response for EMPATHY.
Explain why the generated empathy-augmented response may have been judged less empathetic.
Focus on concrete differences such as warmth, acknowledgement, reassurance, tone, or whether the added wording feels formulaic or inappropriate.
Do not invent facts. Use only the texts below.
Write 1-3 concise sentences.

Patient question:
{question}

Original response:
{original_answer}

Empathy-augmented response:
{generated_answer}

Judge raw answer:
{raw_judge_response}

Explanation:""".strip()
    return system_prompt, user_template


def exact_judge_user_prompt(row: pd.Series, dimension: str) -> str:
    _sys, user_template = judge_prompts(dimension)
    return user_template.format(
        question=normalize_text(row.get("question", "")),
        sentence_a=get_sentence(row, "A"),
        sentence_b=get_sentence(row, "B"),
    )

def parse_ab(raw: str) -> str:
    text = normalize_text(raw).upper()
    if text == "A" or text.startswith("A\n") or text.startswith("A.") or text.startswith("A "):
        return "A"
    if text == "B" or text.startswith("B\n") or text.startswith("B.") or text.startswith("B "):
        return "B"
    m = re.search(r"\b([AB])\b", text)
    return m.group(1) if m else ""



def candidate_csv_paths(path: str) -> List[str]:
    """Return exact path plus common duplicate/download variants like file(1).csv."""
    paths = []
    if path:
        paths.append(path)
        base, ext = os.path.splitext(path)
        for i in range(1, 6):
            paths.append(f"{base}({i}){ext}")
    # keep order, remove duplicates
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def read_csv_robust(path: str) -> pd.DataFrame:
    """Read CSV without crashing the app on a half-written/corrupted result file.

    If the exact file is corrupted, try common duplicate variants. If all exact/variant
    files fail, try the Python parser with bad-line skipping as a last-resort partial
    recovery. This prevents a single interrupted save from hiding all other results.
    """
    first_error = None
    existing = [p for p in candidate_csv_paths(path) if os.path.exists(p)]
    if not existing:
        raise FileNotFoundError(path)

    # First try strict/default parser on exact and variants.
    for p in existing:
        try:
            df = pd.read_csv(p)
            if p != path:
                st.warning(f"Loaded `{os.path.basename(p)}` because `{os.path.basename(path)}` was missing or unreadable.")
            return df
        except Exception as e:
            if first_error is None:
                first_error = e

    # Last resort: recover partial file. This may lose the corrupted/truncated row.
    for p in existing:
        try:
            df = pd.read_csv(p, engine="python", on_bad_lines="skip")
            st.warning(
                f"`{os.path.basename(p)}` appears corrupted or half-written. "
                f"Loaded a partial recovery with {len(df)} rows. Original error: {first_error}"
            )
            return df
        except Exception:
            pass

    raise first_error or RuntimeError(f"Could not read CSV: {path}")


def write_csv_atomic(df: pd.DataFrame, path: str) -> None:
    """Write to a temporary file and atomically replace the target.

    This prevents EOF-inside-string errors caused by Streamlit/API interruptions while
    a CSV is being written.
    """
    tmp_path = f"{path}.tmp"
    df.to_csv(tmp_path, index=False, encoding="utf-8", lineterminator="\n")
    os.replace(tmp_path, path)


def load_dataset(path: str) -> pd.DataFrame:
    df = read_csv_robust(path)
    required = ["id", "general_category", "category", "question", "original_answer"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")
    if "generated_empathy_answer" not in df.columns:
        df["generated_empathy_answer"] = ""
    return df


def initialize_results(dataset_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    df = dataset_df.copy()
    for col in TEXT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Stable balanced A/B assignment AFTER dataset is fixed.
    # Gold means the intended empathy-augmented answer position, not human gold.
    hashes = []
    for idx, row in df.iterrows():
        rid = normalize_text(row.get("id", idx))
        hashes.append((stable_hash_int(seed, rid), idx))
    hashes = sorted(hashes, key=lambda x: x[0])
    half = len(hashes) // 2
    gold_a_indices = {idx for _, idx in hashes[:half]}

    for idx, row in df.iterrows():
        generated_is_a = idx in gold_a_indices
        if generated_is_a:
            df.at[idx, "sentence_a_source"] = "Generated"
            df.at[idx, "sentence_b_source"] = "Original"
            df.at[idx, "gold_choice"] = "A"
            df.at[idx, "response_order"] = "Generated_A__Original_B"
        else:
            df.at[idx, "sentence_a_source"] = "Original"
            df.at[idx, "sentence_b_source"] = "Generated"
            df.at[idx, "gold_choice"] = "B"
            df.at[idx, "response_order"] = "Original_A__Generated_B"
    return coerce_dtypes(df)


def coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(object)
    return df


def blank_generation_outputs(df: pd.DataFrame) -> pd.DataFrame:
    """Clear generated outputs and all downstream judge values."""
    clear_cols = [
        "generated_empathy_answer", "generation_model", "generation_prompt_variant", "generated_at", "generation_error",
        "empathy_pred_choice", "empathy_pred_source", "empathy_raw_judge_response",
        "empathy_judge_model", "empathy_judged_at", "empathy_judge_error",
        "quality_pred_choice", "quality_pred_source", "quality_raw_judge_response",
        "quality_judge_model", "quality_judged_at", "quality_judge_error",
        "empathy_mismatch_explanation", "empathy_mismatch_raw_response", "empathy_mismatch_explained_at", "empathy_mismatch_error",
        "quality_mismatch_explanation", "quality_mismatch_raw_response", "quality_mismatch_explained_at", "quality_mismatch_error",
        "factual_preservation_answer", "factual_preservation_raw_response", "factual_preservation_judged_at", "factual_preservation_error",
        "unsupported_addition_answer", "unsupported_addition_raw_response", "unsupported_addition_judged_at", "unsupported_addition_error",
        "factual_preservation_judge_model", "unsupported_addition_judge_model",
        "pred_choice", "pred_source", "raw_judge_response", "judge_model", "judged_at", "judge_error",
    ]
    for col in clear_cols:
        if col in df.columns:
            df[col] = ""
    return df


def blank_judge_outputs(df: pd.DataFrame) -> pd.DataFrame:
    """Clear only judge/explanation outputs while preserving generated answers."""
    clear_cols = [
        "empathy_pred_choice", "empathy_pred_source", "empathy_raw_judge_response",
        "empathy_judge_model", "empathy_judged_at", "empathy_judge_error",
        "quality_pred_choice", "quality_pred_source", "quality_raw_judge_response",
        "quality_judge_model", "quality_judged_at", "quality_judge_error",
        "empathy_mismatch_explanation", "empathy_mismatch_raw_response", "empathy_mismatch_explained_at", "empathy_mismatch_error",
        "quality_mismatch_explanation", "quality_mismatch_raw_response", "quality_mismatch_explained_at", "quality_mismatch_error",
        "factual_preservation_answer", "factual_preservation_raw_response", "factual_preservation_judged_at", "factual_preservation_error",
        "unsupported_addition_answer", "unsupported_addition_raw_response", "unsupported_addition_judged_at", "unsupported_addition_error",
        "factual_preservation_judge_model", "unsupported_addition_judge_model",
        "pred_choice", "pred_source", "raw_judge_response", "judge_model", "judged_at", "judge_error",
    ]
    for col in clear_cols:
        if col in df.columns:
            df[col] = ""
    return df


def load_or_create_generation_results(dataset_path: str, generation_path: str, seed: int, prompt_variant_key: str = "v1") -> pd.DataFrame:
    """Generation file is model-output source of truth.

    It contains dataset rows, generated_empathy_answer, and the stable A/B assignment.
    It should not contain judge results for different judges. If a legacy combined file
    exists, we migrate generated answers once into the generation file.
    """
    dataset_df = load_dataset(dataset_path)

    source_path = generation_path
    # One-time convenience migration only for the original Llama generation file.
    # Do NOT migrate Llama generations into a GPT-OSS generation file.
    if (not os.path.exists(source_path)
        and os.path.basename(generation_path) == LEGACY_LLAMA_GENERATION_RESULTS
        and os.path.exists(LEGACY_SINGLE_RESULTS)):
        source_path = LEGACY_SINGLE_RESULTS

    if os.path.exists(source_path):
        res = read_csv_robust(source_path)
        res = coerce_dtypes(res)
        res = repair_prompt_variant_from_filename(res, source_path)
        for col in dataset_df.columns:
            if col not in res.columns:
                res[col] = dataset_df[col] if len(dataset_df) == len(res) else ""
        for col in TEXT_COLUMNS:
            if col not in res.columns:
                res[col] = ""
        res = initialize_results(res, seed)
        if "generation_prompt_variant" in res.columns:
            res["generation_prompt_variant"] = res["generation_prompt_variant"].apply(normalize_prompt_variant_value)
        # A generation file belongs to exactly one generation prompt variant.
        # If a file with the same path somehow contains outputs from another variant,
        # do not show them under the current variant. This prevents V1 generations/judgments
        # from appearing when V2 is selected.
        generated_mask = res["generated_empathy_answer"].fillna("").astype(str).str.strip().ne("")
        if generated_mask.any():
            variants = set(res.loc[generated_mask, "generation_prompt_variant"].apply(normalize_prompt_variant_value))
            if variants and variants != {prompt_variant_key}:
                st.warning(
                    f"Generation file `{os.path.basename(generation_path)}` contains outputs for prompt variant(s) "
                    f"{sorted(variants)}, not `{prompt_variant_key}`. Showing this variant as ungenerated."
                )
                res = blank_generation_outputs(res)
        # Remove judge-specific values from the generation master file.
        for col in [
            "empathy_pred_choice", "empathy_pred_source", "empathy_raw_judge_response",
            "empathy_judge_model", "empathy_judged_at", "empathy_judge_error",
            "quality_pred_choice", "quality_pred_source", "quality_raw_judge_response",
            "quality_judge_model", "quality_judged_at", "quality_judge_error",
            "pred_choice", "pred_source", "raw_judge_response", "judge_model", "judged_at", "judge_error",
        ]:
            if col in res.columns:
                res[col] = ""
        return coerce_dtypes(res)

    return initialize_results(dataset_df, seed)


def load_or_create_judge_results(generation_df: pd.DataFrame, judge_path: str, seed: int, prompt_variant_key: str = "v1") -> pd.DataFrame:
    """A separate judge CSV is used for every generator+judge combination.

    Switching either the generator dropdown or the judge dropdown switches this file automatically.
    Generated answers are copied from the selected generation master, but each judge keeps its own
    empathy/quality predictions.
    """
    source_judge_path = judge_path
    # One-time convenience migration for your already completed GPT-OSS judgment on Llama generations.
    if (not os.path.exists(source_judge_path)
        and "GENERATED_BY_llama_JUDGED_BY_plus_openai_gpt_oss_120b" in os.path.basename(judge_path)
        and os.path.exists(LEGACY_LLAMA_GPTOSS_JUDGE_RESULTS)):
        source_judge_path = LEGACY_LLAMA_GPTOSS_JUDGE_RESULTS

    if os.path.exists(source_judge_path):
        res = read_csv_robust(source_judge_path)
        res = coerce_dtypes(res)
        res = repair_prompt_variant_from_filename(res, source_judge_path)
        if "generation_prompt_variant" in res.columns:
            res["generation_prompt_variant"] = res["generation_prompt_variant"].apply(normalize_prompt_variant_value)
        for col in generation_df.columns:
            if col not in res.columns:
                res[col] = generation_df[col] if len(generation_df) == len(res) else ""
        # Always refresh dataset/generation/order columns from generation master.
        refresh_cols = [
            "id", "general_category", "category", "date", "url", "question", "original_answer",
            "question_word_count", "answer_word_count", "generated_empathy_answer",
            "generation_model", "generated_at", "generation_error",
            "response_order", "sentence_a_source", "sentence_b_source", "gold_choice",
        ]
        for col in refresh_cols:
            if col in generation_df.columns:
                res[col] = generation_df[col].values
        for col in TEXT_COLUMNS:
            if col not in res.columns:
                res[col] = ""
        # A judge file is valid only for the currently selected generation prompt variant.
        # If no generated answers exist for this variant, or if the loaded judge file has
        # rows generated by another variant, keep generation rows but blank judge outputs.
        generated_mask = res["generated_empathy_answer"].fillna("").astype(str).str.strip().ne("")
        if not generated_mask.any():
            res = blank_judge_outputs(res)
        else:
            variants = set(res.loc[generated_mask, "generation_prompt_variant"].apply(normalize_prompt_variant_value))
            if variants and variants != {prompt_variant_key}:
                st.warning(
                    f"Judge file `{os.path.basename(judge_path)}` contains outputs for generation prompt variant(s) "
                    f"{sorted(variants)}, not `{prompt_variant_key}`. Judge outputs are hidden for this view."
                )
                res = blank_judge_outputs(res)
        return coerce_dtypes(res)

    res = generation_df.copy()
    # Blank judge columns for this new judge result file.
    for col in [
        "empathy_pred_choice", "empathy_pred_source", "empathy_raw_judge_response",
        "empathy_judge_model", "empathy_judged_at", "empathy_judge_error",
        "quality_pred_choice", "quality_pred_source", "quality_raw_judge_response",
        "quality_judge_model", "quality_judged_at", "quality_judge_error",
        "empathy_mismatch_explanation", "empathy_mismatch_raw_response", "empathy_mismatch_explained_at", "empathy_mismatch_error",
        "quality_mismatch_explanation", "quality_mismatch_raw_response", "quality_mismatch_explained_at", "quality_mismatch_error",
        "factual_preservation_answer", "factual_preservation_raw_response", "factual_preservation_judged_at", "factual_preservation_error",
        "unsupported_addition_answer", "unsupported_addition_raw_response", "unsupported_addition_judged_at", "unsupported_addition_error",
        "factual_preservation_judge_model", "unsupported_addition_judge_model",
        "pred_choice", "pred_source", "raw_judge_response", "judge_model", "judged_at", "judge_error",
    ]:
        if col in res.columns:
            res[col] = ""
    return coerce_dtypes(res)

def save_results(df: pd.DataFrame, path: str) -> None:
    write_csv_atomic(df, path)


def get_sentence(row: pd.Series, letter: str) -> str:
    source = row.get(f"sentence_{letter.lower()}_source", "")
    if source == "Generated":
        return normalize_text(row.get("generated_empathy_answer", ""))
    return normalize_text(row.get("original_answer", ""))


def safe_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    labels = ["Original", "Generated"]
    out = {}
    if not y_true:
        return {"accuracy": np.nan, "macro_precision": np.nan, "macro_recall": np.nan, "macro_f1": np.nan}
    out["accuracy"] = float(np.mean([a == b for a, b in zip(y_true, y_pred)]))
    vals = []
    precs = []
    recs = []
    f1s = []
    for lab in labels:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == lab and b == lab)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != lab and b == lab)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == lab and b != lab)
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f = 2*p*r/(p+r) if (p+r) else 0.0
        precs.append(p); recs.append(r); f1s.append(f)
        out[f"{lab}_precision"] = p
        out[f"{lab}_recall"] = r
        out[f"{lab}_f1"] = f
        out[f"{lab}_support"] = sum(1 for a in y_true if a == lab)
    out["macro_precision"] = float(np.mean(precs))
    out["macro_recall"] = float(np.mean(recs))
    out["macro_f1"] = float(np.mean(f1s))
    return out


st.title("IMB-QA Empathy Generation + Pairwise Judge")
st.caption("Each generator gets its own clean generation CSV; each generator+judge pair gets its own separate judgment CSV automatically.")

with st.sidebar:
    st.header("Files")
    subset_variant = st.selectbox("IMB subset variant", list(DATASET_VARIANTS.keys()), index=0)
    subset_key = DATASET_VARIANTS[subset_variant]["key"]
    default_dataset_for_variant = DATASET_VARIANTS[subset_variant]["path"]
    dataset_path = st.text_input("Dataset CSV path", value=default_dataset_for_variant, key=f"dataset_path_{subset_key}")
    seed = st.number_input("A/B prompt-order seed", value=42, step=1)

    st.header("Generation")
    gen_model_label = st.selectbox("Generator model", GENERATION_MODEL_LABELS, index=0)
    generation_prompt_variant = st.selectbox(
        "Generation prompt variant",
        list(GENERATION_PROMPT_VARIANTS.keys()),
        index=list(GENERATION_PROMPT_VARIANTS.keys()).index("V3: Refined clinical empathy editing prompt"),
        help="V1 keeps the old filenames, so previous results remain visible. V2, V3, and V4 use separate files.",
    )
    generation_prompt_variant_key = GENERATION_PROMPT_VARIANTS[generation_prompt_variant]["key"]
    generation_results_path = generation_results_path_for(gen_model_label, int(seed), subset_key, generation_prompt_variant_key)
    st.text_input("Generation CSV path (auto by subset + prompt variant + generator)", value=generation_results_path, disabled=True)
    gen_base_default = MODEL_DEFAULT_BASE_URL.get(gen_model_label, "")
    gen_api_key = st.text_input("Generator API key", type="password", value=os.environ.get("GEN_API_KEY", os.environ.get("GROQ_API_KEY", "")))
    gen_base_url = st.text_input("Generator Base URL", value=gen_base_default, key=f"gen_base_url_{slugify_label(gen_model_label)}")
    gen_temperature = st.number_input("Generation temperature", min_value=0.0, max_value=1.5, value=0.0, step=0.1)
    gen_max_tokens = st.number_input("Generation max tokens (0 = no explicit limit)", min_value=0, max_value=8192, value=0, step=64)

    active_gen_system_prompt, active_gen_user_prompt_template = generation_prompts(generation_prompt_variant_key)
    with st.expander("Active generation prompt preview", expanded=False):
        st.caption("This is the exact generation prompt template for the selected prompt variant. Placeholders are filled per row during generation.")
        st.text_area("Generation system prompt", value=active_gen_system_prompt, height=80, disabled=True)
        st.text_area("Generation user prompt", value=active_gen_user_prompt_template, height=360, disabled=True)

    st.header("Judge")
    judge_default_idx = JUDGE_MODEL_LABELS.index("+ llama-3.3-70b-versatile") if "+ llama-3.3-70b-versatile" in JUDGE_MODEL_LABELS else 0
    judge_model_label = st.selectbox("Judge model", JUDGE_MODEL_LABELS, index=judge_default_idx)
    judge_results_path = judge_results_path_for(gen_model_label, judge_model_label, int(seed), subset_key, generation_prompt_variant_key)
    st.text_input("Judge CSV path (auto by subset + prompt variant + generator + judge)", value=judge_results_path, disabled=True)
    judge_base_default = MODEL_DEFAULT_BASE_URL.get(judge_model_label, "")
    judge_api_key = st.text_input("Judge API key", type="password", value=os.environ.get("JUDGE_API_KEY", os.environ.get("GROQ_API_KEY", "")))
    judge_base_url = st.text_input("Judge Base URL", value=judge_base_default, key=f"judge_base_url_{slugify_label(judge_model_label)}")
    judge_temperature = st.number_input("Judge temperature", min_value=0.0, max_value=1.5, value=0.0, step=0.1)
    judge_max_tokens = st.number_input("Judge max tokens (0 = no explicit limit)", min_value=0, max_value=1024, value=0, step=8)

    st.header("Batch")
    n_now = st.number_input("Number of rows to run now", min_value=1, max_value=1000, value=100, step=10)
    on_row_error = st.selectbox("On row error", ["Continue", "Stop run after first failure"], index=1)

    st.header("Danger zone")
    st.caption("Deletes only the currently selected auto-file. Use when you want a clean rerun for this exact setting.")
    dc1, dc2 = st.columns(2)
    with dc1:
        if st.button("Delete generation file", key=f"delete_gen_{subset_key}_{generation_prompt_variant_key}_{slugify_label(gen_model_label)}"):
            try:
                if os.path.exists(generation_results_path):
                    os.remove(generation_results_path)
                    st.success(f"Deleted {generation_results_path}")
                else:
                    st.info("Generation file does not exist yet.")
            except Exception as e:
                st.error(f"Could not delete generation file: {e}")
            st.rerun()
    with dc2:
        if st.button("Delete current judge file", key=f"delete_judge_{subset_key}_{generation_prompt_variant_key}_{slugify_label(gen_model_label)}_{slugify_label(judge_model_label)}"):
            try:
                if os.path.exists(judge_results_path):
                    os.remove(judge_results_path)
                    st.success(f"Deleted {judge_results_path}")
                else:
                    st.info("Judge file does not exist yet.")
            except Exception as e:
                st.error(f"Could not delete judge file: {e}")
            st.rerun()

try:
    generation_df = load_or_create_generation_results(dataset_path, generation_results_path, int(seed), generation_prompt_variant_key)
    save_results(generation_df, generation_results_path)
    df = load_or_create_judge_results(generation_df, judge_results_path, int(seed), generation_prompt_variant_key)
    save_results(df, judge_results_path)
except Exception as e:
    st.error(f"Could not load dataset/results: {e}")
    st.stop()

st.info(f"Subset: `{subset_variant}`  |  Generation prompt: `{generation_prompt_variant}`  |  Dataset: `{dataset_path}`")
st.info(f"Generation file: `{generation_results_path}`  |  Current judge file: `{judge_results_path}`")
st.caption("Files are separated automatically by generator and judge. Writes are atomic to avoid half-written CSV corruption.")

generated_mask = df["generated_empathy_answer"].fillna("").astype(str).str.strip().ne("")
empathy_judged_mask = df["empathy_pred_choice"].fillna("").astype(str).str.strip().isin(["A", "B"])
quality_judged_mask = df["quality_pred_choice"].fillna("").astype(str).str.strip().isin(["A", "B"])
judged_mask = empathy_judged_mask & quality_judged_mask
ready_to_judge_mask = generated_mask & ~judged_mask


if generation_prompt_variant_key != "v1" and int(generated_mask.sum()) == 0:
    st.warning("This generation prompt variant has no generated answers yet. Previous V1 judged results are intentionally hidden. Click Generate to create outputs for the selected variant.")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows", len(df))
c2.metric("Generated", int(generated_mask.sum()))
c3.metric("Pending generation", int((~generated_mask).sum()))
c4.metric("Judged both", int(judged_mask.sum()))
c5.metric("Pending judge", int(ready_to_judge_mask.sum()))

st.subheader("Prompt preview")
with st.expander("Generation prompt", expanded=False):
    sys_g, user_g = generation_prompts(generation_prompt_variant_key)
    st.text_area("Generation system prompt", value=sys_g, height=80)
    st.text_area("Generation user prompt template", value=user_g, height=260)
with st.expander("Judge prompts", expanded=False):
    sys_e, user_e = judge_prompts("empathy")
    sys_q, user_q = judge_prompts("quality")
    st.markdown("**Empathy judge prompt**")
    st.text_area("Empathy judge system prompt", value=sys_e, height=80)
    st.text_area("Empathy judge user prompt template", value=user_e, height=220)
    st.markdown("**Quality judge prompt**")
    st.text_area("Quality judge system prompt", value=sys_q, height=80)
    st.text_area("Quality judge user prompt template", value=user_q, height=220)
    sys_fp, user_fp = factual_check_prompts("preservation")
    sys_ua, user_ua = factual_check_prompts("unsupported_addition")
    st.markdown("**Factual preservation check prompt**")
    st.text_area("Factual preservation system prompt", value=sys_fp, height=70)
    st.text_area("Factual preservation user prompt template", value=user_fp, height=220)
    st.markdown("**Unsupported addition check prompt**")
    st.text_area("Unsupported addition system prompt", value=sys_ua, height=70)
    st.text_area("Unsupported addition user prompt template", value=user_ua, height=220)


def save_generation_master_from_current(df_current: pd.DataFrame) -> None:
    gen = df_current.copy()
    for col in [
        "empathy_pred_choice", "empathy_pred_source", "empathy_raw_judge_response",
        "empathy_judge_model", "empathy_judged_at", "empathy_judge_error",
        "quality_pred_choice", "quality_pred_source", "quality_raw_judge_response",
        "quality_judge_model", "quality_judged_at", "quality_judge_error",
        "empathy_mismatch_explanation", "empathy_mismatch_raw_response", "empathy_mismatch_explained_at", "empathy_mismatch_error",
        "quality_mismatch_explanation", "quality_mismatch_raw_response", "quality_mismatch_explained_at", "quality_mismatch_error",
        "factual_preservation_answer", "factual_preservation_raw_response", "factual_preservation_judged_at", "factual_preservation_error",
        "unsupported_addition_answer", "unsupported_addition_raw_response", "unsupported_addition_judged_at", "unsupported_addition_error",
        "factual_preservation_judge_model", "unsupported_addition_judge_model",
        "pred_choice", "pred_source", "raw_judge_response", "judge_model", "judged_at", "judge_error",
    ]:
        if col in gen.columns:
            gen[col] = ""
    save_results(gen, generation_results_path)

# Action buttons
col_a, col_b, col_c, col_d = st.columns(4)

def run_generation(indices: List[int]) -> Dict[str, int]:
    summary = {"attempted": 0, "generated": 0, "failed": 0, "last_failed_id": "", "last_error": ""}
    client = make_client(gen_api_key, gen_base_url)
    model = api_model_name(gen_model_label)
    sys_prompt, user_template = generation_prompts(generation_prompt_variant_key)
    progress = st.progress(0)
    status = st.empty()
    for pos, idx in enumerate(indices, start=1):
        summary["attempted"] += 1
        row = df.loc[idx]
        rid = normalize_text(row.get("id", idx))
        status.write(f"Generating row {rid} ({pos}/{len(indices)})")
        try:
            prompt = user_template.format(
                question=normalize_text(row.get("question", "")),
                original_answer=normalize_text(row.get("original_answer", "")),
            )
            gen_cap = None if int(gen_max_tokens) == 0 else int(gen_max_tokens)
            out = ask_model(client, model, sys_prompt, prompt, temperature=float(gen_temperature), max_tokens=gen_cap)
            df.at[idx, "generated_empathy_answer"] = out
            df.at[idx, "generation_model"] = gen_model_label
            df.at[idx, "generation_prompt_variant"] = generation_prompt_variant_key
            df.at[idx, "generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            df.at[idx, "generation_error"] = ""
            summary["generated"] += 1
        except Exception as e:
            msg = humanize_api_error(e)
            df.at[idx, "generation_error"] = msg
            summary["failed"] += 1
            summary["last_failed_id"] = rid
            summary["last_error"] = msg
            status.error(f"Generation failed on row {rid}: {msg}")
            if on_row_error == "Stop run after first failure":
                break
        save_generation_master_from_current(df)
        save_results(df, judge_results_path)
        progress.progress(pos / max(1, len(indices)))
    return summary


def run_single_judge_for_dimension(client: OpenAI, model: str, row: pd.Series, dimension: str) -> Tuple[str, str, str, str]:
    sys_prompt, user_template = judge_prompts(dimension)
    sentence_a = get_sentence(row, "A")
    sentence_b = get_sentence(row, "B")
    prompt = user_template.format(
        question=normalize_text(row.get("question", "")),
        sentence_a=sentence_a,
        sentence_b=sentence_b,
    )
    judge_cap = None if int(judge_max_tokens) == 0 else int(judge_max_tokens)
    raw = ask_model(client, model, sys_prompt, prompt, temperature=float(judge_temperature), max_tokens=judge_cap)
    pred_choice = parse_ab(raw)
    pred_source = ""
    if pred_choice == "A":
        pred_source = normalize_text(row.get("sentence_a_source", ""))
    elif pred_choice == "B":
        pred_source = normalize_text(row.get("sentence_b_source", ""))
    err = "" if pred_choice in ["A", "B"] else "Could not parse A/B"
    return raw, pred_choice, pred_source, err


def run_judge(indices: List[int]) -> Dict[str, int]:
    summary = {"attempted_rows": 0, "empathy_judged": 0, "quality_judged": 0, "failed": 0}
    client = make_client(judge_api_key, judge_base_url)
    model = api_model_name(judge_model_label)
    progress = st.progress(0)
    status = st.empty()
    for pos, idx in enumerate(indices, start=1):
        summary["attempted_rows"] += 1
        row = df.loc[idx]
        rid = normalize_text(row.get("id", idx))
        status.write(f"Judging row {rid} ({pos}/{len(indices)}): empathy + quality")
        try:
            now = time.strftime("%Y-%m-%d %H:%M:%S")

            if normalize_text(row.get("empathy_pred_choice", "")) not in ["A", "B"]:
                raw, pred_choice, pred_source, err = run_single_judge_for_dimension(client, model, row, "empathy")
                df.at[idx, "empathy_raw_judge_response"] = raw
                df.at[idx, "empathy_pred_choice"] = pred_choice
                df.at[idx, "empathy_pred_source"] = pred_source
                df.at[idx, "empathy_judge_model"] = judge_model_label
                df.at[idx, "empathy_judged_at"] = now
                df.at[idx, "empathy_judge_error"] = err
                if pred_choice in ["A", "B"]:
                    summary["empathy_judged"] += 1
                else:
                    summary["failed"] += 1
                    if on_row_error == "Stop run after first failure":
                        save_results(df, judge_results_path)
                        break

            if normalize_text(row.get("quality_pred_choice", "")) not in ["A", "B"]:
                # Refresh row after empathy update but same A/B/order/source content.
                row = df.loc[idx]
                raw, pred_choice, pred_source, err = run_single_judge_for_dimension(client, model, row, "quality")
                df.at[idx, "quality_raw_judge_response"] = raw
                df.at[idx, "quality_pred_choice"] = pred_choice
                df.at[idx, "quality_pred_source"] = pred_source
                df.at[idx, "quality_judge_model"] = judge_model_label
                df.at[idx, "quality_judged_at"] = now
                df.at[idx, "quality_judge_error"] = err
                if pred_choice in ["A", "B"]:
                    summary["quality_judged"] += 1
                else:
                    summary["failed"] += 1
                    if on_row_error == "Stop run after first failure":
                        save_results(df, judge_results_path)
                        break

        except Exception as e:
            msg = str(e)
            if normalize_text(df.at[idx, "empathy_pred_choice"]) not in ["A", "B"]:
                df.at[idx, "empathy_judge_error"] = msg
            if normalize_text(df.at[idx, "quality_pred_choice"]) not in ["A", "B"]:
                df.at[idx, "quality_judge_error"] = msg
            summary["failed"] += 1
            summary["last_failed_id"] = rid
            summary["last_error"] = msg
            if on_row_error == "Stop run after first failure":
                save_results(df, judge_results_path)
                break
        save_generation_master_from_current(df)
        save_results(df, judge_results_path)
        progress.progress(pos / max(1, len(indices)))
    return summary



def run_single_factual_check(client: OpenAI, model: str, row: pd.Series, check_type: str) -> Tuple[str, str, str]:
    sys_prompt, user_template = factual_check_prompts(check_type)
    prompt = user_template.format(
        question=normalize_text(row.get("question", "")),
        original_answer=normalize_text(row.get("original_answer", "")),
        generated_answer=normalize_text(row.get("generated_empathy_answer", "")),
    )
    judge_cap = None if int(judge_max_tokens) == 0 else int(judge_max_tokens)
    raw = ask_model(client, model, sys_prompt, prompt, temperature=float(judge_temperature), max_tokens=judge_cap)
    ans = parse_yes_no(raw)
    err = "" if ans in ["Yes", "No"] else "Could not parse Yes/No"
    return raw, ans, err


def run_factual_checks(indices: List[int]) -> Dict[str, int]:
    summary = {"attempted_rows": 0, "preservation_checked": 0, "unsupported_checked": 0, "failed": 0}
    client = make_client(judge_api_key, judge_base_url)
    model = api_model_name(judge_model_label)
    progress = st.progress(0)
    status = st.empty()
    for pos, idx in enumerate(indices, start=1):
        summary["attempted_rows"] += 1
        row = df.loc[idx]
        rid = normalize_text(row.get("id", idx))
        status.write(f"Factual checks row {rid} ({pos}/{len(indices)}): preservation + unsupported additions")
        try:
            now = time.strftime("%Y-%m-%d %H:%M:%S")

            if normalize_text(row.get("factual_preservation_answer", "")) not in ["Yes", "No"]:
                raw, ans, err = run_single_factual_check(client, model, row, "preservation")
                df.at[idx, "factual_preservation_raw_response"] = raw
                df.at[idx, "factual_preservation_answer"] = ans
                df.at[idx, "factual_preservation_judge_model"] = judge_model_label
                df.at[idx, "factual_preservation_judged_at"] = now
                df.at[idx, "factual_preservation_error"] = err
                if ans in ["Yes", "No"]:
                    summary["preservation_checked"] += 1
                else:
                    summary["failed"] += 1
                    if on_row_error == "Stop run after first failure":
                        save_results(df, judge_results_path)
                        break

            row = df.loc[idx]
            if normalize_text(row.get("unsupported_addition_answer", "")) not in ["Yes", "No"]:
                raw, ans, err = run_single_factual_check(client, model, row, "unsupported_addition")
                df.at[idx, "unsupported_addition_raw_response"] = raw
                df.at[idx, "unsupported_addition_answer"] = ans
                df.at[idx, "unsupported_addition_judge_model"] = judge_model_label
                df.at[idx, "unsupported_addition_judged_at"] = now
                df.at[idx, "unsupported_addition_error"] = err
                if ans in ["Yes", "No"]:
                    summary["unsupported_checked"] += 1
                else:
                    summary["failed"] += 1
                    if on_row_error == "Stop run after first failure":
                        save_results(df, judge_results_path)
                        break
        except Exception as e:
            msg = str(e)
            if normalize_text(df.at[idx, "factual_preservation_answer"]) not in ["Yes", "No"]:
                df.at[idx, "factual_preservation_error"] = msg
            if normalize_text(df.at[idx, "unsupported_addition_answer"]) not in ["Yes", "No"]:
                df.at[idx, "unsupported_addition_error"] = msg
            summary["failed"] += 1
            if on_row_error == "Stop run after first failure":
                save_results(df, judge_results_path)
                break
        save_results(df, judge_results_path)
        progress.progress(pos / max(1, len(indices)))
    return summary


def factual_pending_indices() -> List[int]:
    mask = df["generated_empathy_answer"].fillna("").astype(str).str.strip().ne("") & (
        ~df["factual_preservation_answer"].fillna("").astype(str).isin(["Yes", "No"])
        | ~df["unsupported_addition_answer"].fillna("").astype(str).isin(["Yes", "No"])
    )
    return df.index[mask].tolist()



def run_judge_all_unified(indices: List[int]) -> Dict[str, int]:
    """
    Run all pending judgments for the SAME selected rows in one unified loop.
    This avoids the confusing behavior where preference judging and factual checks
    use different progress denominators or appear to jump to a different row set.
    For each selected row, the function runs only the missing prompts:
    empathy, quality, factual preservation, unsupported additions.
    """
    summary = {
        "attempted_rows": 0,
        "empathy_judged": 0,
        "quality_judged": 0,
        "preservation_checked": 0,
        "unsupported_checked": 0,
        "failed": 0,
        "last_failed_id": "",
        "last_error": "",
    }
    client = make_client(judge_api_key, judge_base_url)
    model = api_model_name(judge_model_label)
    progress = st.progress(0)
    status = st.empty()

    for pos, idx in enumerate(indices, start=1):
        summary["attempted_rows"] += 1
        row = df.loc[idx]
        rid = normalize_text(row.get("id", idx))
        status.write(f"Judge-all row {rid} ({pos}/{len(indices)}): empathy + quality + factual checks")
        try:
            now = time.strftime("%Y-%m-%d %H:%M:%S")

            # 1) Empathy preference
            row = df.loc[idx]
            if normalize_text(row.get("empathy_pred_choice", "")) not in ["A", "B"]:
                raw, pred_choice, pred_source, err = run_single_judge_for_dimension(client, model, row, "empathy")
                df.at[idx, "empathy_raw_judge_response"] = raw
                df.at[idx, "empathy_pred_choice"] = pred_choice
                df.at[idx, "empathy_pred_source"] = pred_source
                df.at[idx, "empathy_judge_model"] = judge_model_label
                df.at[idx, "empathy_judged_at"] = now
                df.at[idx, "empathy_judge_error"] = err
                if pred_choice in ["A", "B"]:
                    summary["empathy_judged"] += 1
                else:
                    summary["failed"] += 1
                    if on_row_error == "Stop run after first failure":
                        save_generation_master_from_current(df)
                        save_results(df, judge_results_path)
                        break

            # 2) Broad quality preference
            row = df.loc[idx]
            if normalize_text(row.get("quality_pred_choice", "")) not in ["A", "B"]:
                raw, pred_choice, pred_source, err = run_single_judge_for_dimension(client, model, row, "quality")
                df.at[idx, "quality_raw_judge_response"] = raw
                df.at[idx, "quality_pred_choice"] = pred_choice
                df.at[idx, "quality_pred_source"] = pred_source
                df.at[idx, "quality_judge_model"] = judge_model_label
                df.at[idx, "quality_judged_at"] = now
                df.at[idx, "quality_judge_error"] = err
                if pred_choice in ["A", "B"]:
                    summary["quality_judged"] += 1
                else:
                    summary["failed"] += 1
                    if on_row_error == "Stop run after first failure":
                        save_generation_master_from_current(df)
                        save_results(df, judge_results_path)
                        break

            # 3) Factual preservation
            row = df.loc[idx]
            if normalize_text(row.get("factual_preservation_answer", "")) not in ["Yes", "No"]:
                raw, ans, err = run_single_factual_check(client, model, row, "preservation")
                df.at[idx, "factual_preservation_raw_response"] = raw
                df.at[idx, "factual_preservation_answer"] = ans
                df.at[idx, "factual_preservation_judge_model"] = judge_model_label
                df.at[idx, "factual_preservation_judged_at"] = now
                df.at[idx, "factual_preservation_error"] = err
                if ans in ["Yes", "No"]:
                    summary["preservation_checked"] += 1
                else:
                    summary["failed"] += 1
                    if on_row_error == "Stop run after first failure":
                        save_generation_master_from_current(df)
                        save_results(df, judge_results_path)
                        break

            # 4) Unsupported additions
            row = df.loc[idx]
            if normalize_text(row.get("unsupported_addition_answer", "")) not in ["Yes", "No"]:
                raw, ans, err = run_single_factual_check(client, model, row, "unsupported_addition")
                df.at[idx, "unsupported_addition_raw_response"] = raw
                df.at[idx, "unsupported_addition_answer"] = ans
                df.at[idx, "unsupported_addition_judge_model"] = judge_model_label
                df.at[idx, "unsupported_addition_judged_at"] = now
                df.at[idx, "unsupported_addition_error"] = err
                if ans in ["Yes", "No"]:
                    summary["unsupported_checked"] += 1
                else:
                    summary["failed"] += 1
                    if on_row_error == "Stop run after first failure":
                        save_generation_master_from_current(df)
                        save_results(df, judge_results_path)
                        break

        except Exception as e:
            msg = humanize_api_error(e)
            status.error(f"Judge-all failed on row {rid}: {msg}")
            # Store the error in any still-pending columns so it is visible in Inspect.
            row = df.loc[idx]
            if normalize_text(row.get("empathy_pred_choice", "")) not in ["A", "B"]:
                df.at[idx, "empathy_judge_error"] = msg
            if normalize_text(row.get("quality_pred_choice", "")) not in ["A", "B"]:
                df.at[idx, "quality_judge_error"] = msg
            if normalize_text(row.get("factual_preservation_answer", "")) not in ["Yes", "No"]:
                df.at[idx, "factual_preservation_error"] = msg
            if normalize_text(row.get("unsupported_addition_answer", "")) not in ["Yes", "No"]:
                df.at[idx, "unsupported_addition_error"] = msg
            summary["failed"] += 1
            if on_row_error == "Stop run after first failure":
                save_generation_master_from_current(df)
                save_results(df, judge_results_path)
                break

        save_generation_master_from_current(df)
        save_results(df, judge_results_path)
        progress.progress(pos / max(1, len(indices)))

    return summary



def mismatch_explanation_pending_indices() -> List[int]:
    pending = []
    for idx, row in df.iterrows():
        empathy_needs = (
            normalize_text(row.get("empathy_pred_source", "")) == "Original"
            and normalize_text(row.get("empathy_mismatch_explanation", "")) == ""
            and normalize_text(row.get("empathy_mismatch_error", "")) == ""
        )
        quality_needs = (
            normalize_text(row.get("quality_pred_source", "")) == "Original"
            and normalize_text(row.get("quality_mismatch_explanation", "")) == ""
            and normalize_text(row.get("quality_mismatch_error", "")) == ""
        )
        if empathy_needs or quality_needs:
            pending.append(idx)
    return pending


def run_mismatch_explanations(indices: List[int]) -> Dict[str, int]:
    summary = {"attempted_rows": 0, "empathy_explained": 0, "quality_explained": 0, "failed": 0}
    client = make_client(judge_api_key, judge_base_url)
    model = api_model_name(judge_model_label)
    progress = st.progress(0)
    status = st.empty()
    for pos, idx in enumerate(indices, start=1):
        summary["attempted_rows"] += 1
        row = df.loc[idx]
        rid = normalize_text(row.get("id", idx))
        status.write(f"Explaining mismatch row {rid} ({pos}/{len(indices)})")
        try:
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            for dim in ["empathy", "quality"]:
                pred_source_col = f"{dim}_pred_source"
                explanation_col = f"{dim}_mismatch_explanation"
                raw_col = f"{dim}_mismatch_raw_response"
                at_col = f"{dim}_mismatch_explained_at"
                err_col = f"{dim}_mismatch_error"
                judge_raw_col = f"{dim}_raw_judge_response"
                if normalize_text(row.get(pred_source_col, "")) != "Original":
                    continue
                if normalize_text(row.get(explanation_col, "")) or normalize_text(row.get(err_col, "")):
                    continue
                sys_prompt, user_template = mismatch_explanation_prompts(dim)
                prompt = user_template.format(
                    question=normalize_text(row.get("question", "")),
                    original_answer=normalize_text(row.get("original_answer", "")),
                    generated_answer=normalize_text(row.get("generated_empathy_answer", "")),
                    raw_judge_response=normalize_text(row.get(judge_raw_col, "")),
                )
                judge_cap = None if int(judge_max_tokens) == 0 else int(judge_max_tokens)
                raw = ask_model(client, model, sys_prompt, prompt, temperature=float(judge_temperature), max_tokens=judge_cap)
                df.at[idx, raw_col] = raw
                df.at[idx, explanation_col] = raw
                df.at[idx, at_col] = now
                df.at[idx, err_col] = ""
                summary[f"{dim}_explained"] += 1
        except Exception as e:
            msg = str(e)
            # Store the error in both potential explanation slots for visibility.
            if normalize_text(row.get("empathy_pred_source", "")) == "Original" and not normalize_text(row.get("empathy_mismatch_explanation", "")):
                df.at[idx, "empathy_mismatch_error"] = msg
            if normalize_text(row.get("quality_pred_source", "")) == "Original" and not normalize_text(row.get("quality_mismatch_explanation", "")):
                df.at[idx, "quality_mismatch_error"] = msg
            summary["failed"] += 1
            if on_row_error == "Stop run after first failure":
                save_results(df, judge_results_path)
                break
        save_results(df, judge_results_path)
        progress.progress(pos / max(1, len(indices)))
    return summary

with col_a:
    if st.button(f"Generate next {int(n_now)} pending rows", type="primary"):
        pending = df.index[~generated_mask].tolist()[: int(n_now)]
        if not pending:
            st.info("No pending generation rows.")
        elif not gen_api_key:
            st.error("Generator API key is empty.")
        else:
            summary = run_generation(pending)
            if summary.get("failed", 0):
                st.error(f"Generation stopped/failed. Summary: {summary}")
                if summary.get("last_error"):
                    st.error(summary["last_error"])
            else:
                st.success(f"Generation summary: {summary}")
                st.rerun()

with col_b:
    # One-click full judging: this runs the same two preference prompts as before
    # (empathy + broad quality), then the two factual checks (preservation + unsupported additions).
    if st.button(f"Judge all next {int(n_now)} generated pending rows", type="primary"):
        generated_ready = df["generated_empathy_answer"].fillna("").astype(str).str.strip().ne("")
        pref_pending_mask = generated_ready & (
            ~df["empathy_pred_choice"].fillna("").astype(str).isin(["A", "B"])
            | ~df["quality_pred_choice"].fillna("").astype(str).isin(["A", "B"])
        )
        fact_pending_mask = generated_ready & (
            ~df["factual_preservation_answer"].fillna("").astype(str).isin(["Yes", "No"])
            | ~df["unsupported_addition_answer"].fillna("").astype(str).isin(["Yes", "No"])
        )
        combined_pending = list(dict.fromkeys(df.index[pref_pending_mask | fact_pending_mask].tolist()))[: int(n_now)]
        if not combined_pending:
            st.info("No generated rows pending preference/factual judgment.")
        elif not judge_api_key:
            st.error("Judge API key is empty.")
        else:
            summary = run_judge_all_unified(combined_pending)
            if summary.get("failed", 0):
                st.error(f"Judge-all stopped/failed. Summary: {summary}")
                if summary.get("last_error"):
                    st.error(summary["last_error"])
            else:
                st.success(f"Judge-all summary: {summary}")
                st.rerun()

with col_c:
    with st.expander("Advanced: run only preference or only factual checks"):
        if st.button(f"Preference only: judge next {int(n_now)} generated pending rows"):
            mask = df["generated_empathy_answer"].fillna("").astype(str).str.strip().ne("") & (~df["empathy_pred_choice"].fillna("").astype(str).isin(["A", "B"]) | ~df["quality_pred_choice"].fillna("").astype(str).isin(["A", "B"]))
            pending = df.index[mask].tolist()[: int(n_now)]
            if not pending:
                st.info("No generated rows pending preference judgment.")
            elif not judge_api_key:
                st.error("Judge API key is empty.")
            else:
                summary = run_judge(pending)
                st.success(f"Preference judge summary: {summary}")
                st.rerun()

        pending_fact = factual_pending_indices()
        if st.button(f"Factual only: check next {min(int(n_now), len(pending_fact))} generated rows"):
            if not pending_fact:
                st.info("No generated rows pending factual checks.")
            elif not judge_api_key:
                st.error("Judge API key is empty.")
            else:
                summary = run_factual_checks(pending_fact[: int(n_now)])
                st.success(f"Factual check summary: {summary}")
                st.rerun()

with col_d:
    pending_expl = mismatch_explanation_pending_indices()
    if st.button(f"Explain next {min(int(n_now), len(pending_expl))} generated-not-preferred rows"):
        if not pending_expl:
            st.info("No rows need mismatch explanations. This only applies when the judge preferred Original over Generated.")
        elif not judge_api_key:
            st.error("Judge API key is empty.")
        else:
            summary = run_mismatch_explanations(pending_expl[: int(n_now)])
            st.success(f"Mismatch explanation summary: {summary}")
            st.rerun()

st.markdown("---")
st.markdown("**Downloads**")
with st.container():
    st.download_button(
        "Download current judge results CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=os.path.basename(judge_results_path),
        mime="text/csv",
    )
    st.download_button(
        "Download generation CSV",
        data=read_csv_robust(generation_results_path).to_csv(index=False).encode("utf-8") if os.path.exists(generation_results_path) else df.to_csv(index=False).encode("utf-8"),
        file_name=os.path.basename(generation_results_path),
        mime="text/csv",
    )

st.divider()

# Dashboard metrics
def render_dimension_dashboard(label: str, pred_choice_col: str, pred_source_col: str, raw_col: str, error_col: str) -> None:
    scored_dim = df[df[pred_choice_col].fillna("").astype(str).isin(["A", "B"])].copy()
    st.subheader(f"{label} judge preference results")
    if scored_dim.empty:
        st.info(f"No {label.lower()} judgments yet.")
        return

    y_pred = scored_dim[pred_source_col].fillna("").astype(str).tolist()
    pref_generated = sum(1 for x in y_pred if x == "Generated")
    pref_original = sum(1 for x in y_pred if x == "Original")
    failed = int(scored_dim[error_col].fillna("").astype(str).str.strip().ne("").sum())

    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Rows judged", len(scored_dim))
    d2.metric("Generated preferred", pref_generated)
    d3.metric("Original preferred", pref_original)
    d4.metric("Generated win rate", f"{pref_generated / len(scored_dim):.3f}")
    d5.metric("Failed/parse errors", failed)

    st.markdown(f"**{label} Original/Generated source preference matrix**")
    st.caption(
        "IMB has no external human gold label. This matrix treats the Generated answer as the target "
        "for the empathy-augmentation hypothesis and shows whether the judge preferred Original or Generated, "
        "after converting A/B back to source labels."
    )
    source_mat = pd.DataFrame(
        [[pref_original, pref_generated]],
        index=["Target: Generated"],
        columns=["Pred Original", "Pred Generated"],
    )
    st.dataframe(source_mat, use_container_width=True)
    fig_source = px.imshow(
        source_mat,
        text_auto=True,
        aspect="auto",
        labels=dict(x="Judge preference", y="Target/source", color="Count"),
        title=f"{label}: Original vs Generated preference",
    )
    st.plotly_chart(fig_source, use_container_width=True, key=f"source_pref_{label}")

    # Position diagnostic: A/B is randomized. This should not be used as the substantive source result.
    ab_mat = pd.crosstab(scored_dim["gold_choice"], scored_dim[pred_choice_col]).reindex(index=["B", "A"], columns=["A", "B"], fill_value=0)
    st.markdown(f"**{label} A/B diagnostic matrix**")
    st.caption("This checks position behavior only. Top row means Generated was B; bottom row means Generated was A.")
    st.dataframe(ab_mat, use_container_width=True)
    fig = px.imshow(ab_mat, text_auto=True, aspect="auto", labels=dict(x="Predicted letter", y="Generated answer position", color="Count"), title=f"{label}: A/B position diagnostic")
    st.plotly_chart(fig, use_container_width=True, key=f"ab_diag_{label}")

    # Source-by-position diagnostic: useful for seeing whether source preference changes when Generated is A vs B.
    src_by_pos = pd.crosstab(scored_dim["response_order"], scored_dim[pred_source_col]).reindex(
        index=["Original_A__Generated_B", "Generated_A__Original_B"],
        columns=["Original", "Generated"],
        fill_value=0,
    )
    st.markdown(f"**{label} source preference by randomized order**")
    st.caption("This shows whether the judge still prefers Generated when it appears as A vs when it appears as B.")
    st.dataframe(src_by_pos.rename(columns={"Original": "Pred Original", "Generated": "Pred Generated"}), use_container_width=True)

render_dimension_dashboard("Empathy", "empathy_pred_choice", "empathy_pred_source", "empathy_raw_judge_response", "empathy_judge_error")
render_dimension_dashboard("Quality", "quality_pred_choice", "quality_pred_source", "quality_raw_judge_response", "quality_judge_error")

st.subheader("Factual preservation checks")
fact_done = df[df["factual_preservation_answer"].fillna("").astype(str).isin(["Yes", "No"]) & df["unsupported_addition_answer"].fillna("").astype(str).isin(["Yes", "No"])].copy()
if fact_done.empty:
    st.info("No factual preservation checks yet. Use the Factual-check button after generation.")
else:
    preserved = int((fact_done["factual_preservation_answer"].astype(str) == "Yes").sum())
    not_preserved = int((fact_done["factual_preservation_answer"].astype(str) == "No").sum())
    unsupported_yes = int((fact_done["unsupported_addition_answer"].astype(str) == "Yes").sum())
    unsupported_no = int((fact_done["unsupported_addition_answer"].astype(str) == "No").sum())
    factual_pass = int(((fact_done["factual_preservation_answer"].astype(str) == "Yes") & (fact_done["unsupported_addition_answer"].astype(str) == "No")).sum())
    f1, f2, f3, f4, f5 = st.columns(5)
    f1.metric("Rows checked", len(fact_done))
    f2.metric("Preserved info = Yes", preserved)
    f3.metric("Unsupported additions = Yes", unsupported_yes)
    f4.metric("Factual pass", factual_pass)
    f5.metric("Factual pass rate", f"{factual_pass / len(fact_done):.3f}")
    # Factual outcome matrix: Preservation is the y-axis, unsupported additions is the x-axis.
    fact_mat = pd.crosstab(
        fact_done["factual_preservation_answer"].map({"Yes": "Preserved", "No": "Not preserved"}),
        fact_done["unsupported_addition_answer"].map({"No": "No unsupported additions", "Yes": "Unsupported additions"}),
    ).reindex(
        index=["Preserved", "Not preserved"],
        columns=["No unsupported additions", "Unsupported additions"],
        fill_value=0,
    )

    st.markdown("**Factual outcome matrix: preservation × unsupported additions**")
    st.caption(
        "Best case is the top-left cell: the generated answer preserves clinically important information "
        "and does not introduce unsupported medical content. This is the factual-pass cell."
    )

    cmat1, cmat2 = st.columns([1.05, 1.0])
    with cmat1:
        st.dataframe(fact_mat, use_container_width=True)
    with cmat2:
        fig_fact = px.imshow(
            fact_mat,
            text_auto=True,
            aspect="auto",
            labels=dict(x="Unsupported additions check", y="Preservation check", color="Rows"),
            title="Factual checks: preservation × additions",
        )
        st.plotly_chart(fig_fact, use_container_width=True, key="factual_outcome_matrix")

    factual_interpretation = pd.DataFrame([
        {
            "Preservation": "Yes",
            "Unsupported additions": "No",
            "Interpretation": "Factual pass: information preserved and no unsupported medical additions.",
            "Use in paper": "Safe / factual-quality preserved",
        },
        {
            "Preservation": "Yes",
            "Unsupported additions": "Yes",
            "Interpretation": "Preserved original information, but added unsupported medical content.",
            "Use in paper": "Unsafe/uncertain quality despite preference",
        },
        {
            "Preservation": "No",
            "Unsupported additions": "No",
            "Interpretation": "No unsupported additions, but some clinically important original information was lost.",
            "Use in paper": "Information-loss failure",
        },
        {
            "Preservation": "No",
            "Unsupported additions": "Yes",
            "Interpretation": "Both information loss and unsupported additions detected.",
            "Use in paper": "Worst factual-quality failure",
        },
    ])
    with st.expander("How to read this factual matrix", expanded=False):
        st.dataframe(factual_interpretation, use_container_width=True, hide_index=True)
        st.markdown(
            "**Factual pass = Preservation Yes + Unsupported additions No.** "
            "This means the generated answer is factually acceptable relative to the original. "
            "It does not by itself mean the generated answer is broadly better; for that, combine it with the broad quality preference."
        )

    # Composite interpretation: broad quality preference + factual pass.
    # This should be treated as a conservative derived outcome, not as a separate LLM judgment.
    fact_done["factual_pass"] = (
        (fact_done["factual_preservation_answer"].astype(str) == "Yes")
        & (fact_done["unsupported_addition_answer"].astype(str) == "No")
    )
    fact_done["quality_pref_generated"] = fact_done["quality_pred_source"].fillna("").astype(str).eq("Generated")
    fact_done["empathy_pref_generated"] = fact_done["empathy_pred_source"].fillna("").astype(str).eq("Generated")
    fact_done["quality_safe_improvement"] = fact_done["quality_pref_generated"] & fact_done["factual_pass"]
    fact_done["empathy_safe_improvement"] = fact_done["empathy_pref_generated"] & fact_done["factual_pass"]

    cqi1, cqi2, cqi3, cqi4 = st.columns(4)
    cqi1.metric("Quality pref Generated + factual pass", int(fact_done["quality_safe_improvement"].sum()))
    cqi2.metric("Rate", f'{fact_done["quality_safe_improvement"].mean():.3f}')
    cqi3.metric("Empathy pref Generated + factual pass", int(fact_done["empathy_safe_improvement"].sum()))
    cqi4.metric("Rate", f'{fact_done["empathy_safe_improvement"].mean():.3f}')

    composite_mat = pd.crosstab(
        fact_done["quality_pref_generated"].map({True: "Generated preferred", False: "Original/no generated preference"}),
        fact_done["factual_pass"].map({True: "Factual pass", False: "Factual fail"}),
    ).reindex(
        index=["Generated preferred", "Original/no generated preference"],
        columns=["Factual pass", "Factual fail"],
        fill_value=0,
    )
    st.markdown("**Composite quality interpretation matrix: broad quality preference × factual pass**")
    st.caption(
        "The strongest conservative quality claim is the top-left cell: the judge prefers the generated answer for broad quality "
        "AND the generated answer passes factual checks. If quality prefers Generated but factual checks fail, treat it as unsafe/uncertain rather than clean improvement."
    )
    ccomp1, ccomp2 = st.columns([1.05, 1.0])
    with ccomp1:
        st.dataframe(composite_mat, use_container_width=True)
    with ccomp2:
        fig_comp = px.imshow(
            composite_mat,
            text_auto=True,
            aspect="auto",
            labels=dict(x="Factual status", y="Broad quality preference", color="Rows"),
            title="Composite quality interpretation",
        )
        st.plotly_chart(fig_comp, use_container_width=True, key="composite_quality_matrix")

    st.markdown(
        "**Derived claim rule:** safe quality improvement = broad quality prefers Generated **and** factual pass. "
        "If only one of the two is true, do not call it clean quality improvement."
    )

st.caption("For IMB there is no external human gold label. Preference results show whether the judge prefers the generated empathy-augmented answer or the original answer. Factual checks evaluate preservation and unsupported additions separately from broad quality.")

st.subheader("Inspect rows")

filter_option = st.selectbox(
    "Rows to show",
    [
        "All",
        "Pending generation",
        "Generated not judged for both",
        "Judged for both",
        "Empathy preferred Generated",
        "Empathy preferred Original",
        "Quality preferred Generated",
        "Quality preferred Original",
        "Needs mismatch explanation",
        "Has mismatch explanation",
        "Generation errors",
        "Empathy judge errors",
        "Quality judge errors",
        "Factual checks complete",
        "Factual pass",
        "Factual fail",
        "Preservation failed",
        "Unsupported additions",
        "Factual check errors",
        "Mismatch explanation errors",
    ],
    index=0,
)

view = df.copy()
if filter_option == "Pending generation":
    view = view[view["generated_empathy_answer"].fillna("").astype(str).str.strip().eq("")]
elif filter_option == "Generated not judged for both":
    view = view[
        view["generated_empathy_answer"].fillna("").astype(str).str.strip().ne("")
        & (~view["empathy_pred_choice"].fillna("").astype(str).isin(["A", "B"]) | ~view["quality_pred_choice"].fillna("").astype(str).isin(["A", "B"]))
    ]
elif filter_option == "Judged for both":
    view = view[view["empathy_pred_choice"].fillna("").astype(str).isin(["A", "B"]) & view["quality_pred_choice"].fillna("").astype(str).isin(["A", "B"])]
elif filter_option == "Empathy preferred Generated":
    view = view[view["empathy_pred_source"].fillna("").astype(str).eq("Generated")]
elif filter_option == "Empathy preferred Original":
    view = view[view["empathy_pred_source"].fillna("").astype(str).eq("Original")]
elif filter_option == "Quality preferred Generated":
    view = view[view["quality_pred_source"].fillna("").astype(str).eq("Generated")]
elif filter_option == "Quality preferred Original":
    view = view[view["quality_pred_source"].fillna("").astype(str).eq("Original")]
elif filter_option == "Needs mismatch explanation":
    view = view[
        ((view["empathy_pred_source"].fillna("").astype(str).eq("Original")) & (view["empathy_mismatch_explanation"].fillna("").astype(str).str.strip().eq("")) & (view["empathy_mismatch_error"].fillna("").astype(str).str.strip().eq("")))
        | ((view["quality_pred_source"].fillna("").astype(str).eq("Original")) & (view["quality_mismatch_explanation"].fillna("").astype(str).str.strip().eq("")) & (view["quality_mismatch_error"].fillna("").astype(str).str.strip().eq("")))
    ]
elif filter_option == "Has mismatch explanation":
    view = view[
        view["empathy_mismatch_explanation"].fillna("").astype(str).str.strip().ne("")
        | view["quality_mismatch_explanation"].fillna("").astype(str).str.strip().ne("")
    ]
elif filter_option == "Generation errors":
    view = view[view["generation_error"].fillna("").astype(str).str.strip().ne("")]
elif filter_option == "Empathy judge errors":
    view = view[view["empathy_judge_error"].fillna("").astype(str).str.strip().ne("")]
elif filter_option == "Quality judge errors":
    view = view[view["quality_judge_error"].fillna("").astype(str).str.strip().ne("")]
elif filter_option == "Factual checks complete":
    view = view[view["factual_preservation_answer"].fillna("").astype(str).isin(["Yes", "No"]) & view["unsupported_addition_answer"].fillna("").astype(str).isin(["Yes", "No"])]
elif filter_option == "Factual pass":
    view = view[(view["factual_preservation_answer"].fillna("").astype(str) == "Yes") & (view["unsupported_addition_answer"].fillna("").astype(str) == "No")]
elif filter_option == "Factual fail":
    view = view[view["factual_preservation_answer"].fillna("").astype(str).isin(["Yes", "No"]) & view["unsupported_addition_answer"].fillna("").astype(str).isin(["Yes", "No"]) & ~((view["factual_preservation_answer"].fillna("").astype(str) == "Yes") & (view["unsupported_addition_answer"].fillna("").astype(str) == "No"))]
elif filter_option == "Preservation failed":
    view = view[view["factual_preservation_answer"].fillna("").astype(str) == "No"]
elif filter_option == "Unsupported additions":
    view = view[view["unsupported_addition_answer"].fillna("").astype(str) == "Yes"]
elif filter_option == "Factual check errors":
    view = view[view["factual_preservation_error"].fillna("").astype(str).str.strip().ne("") | view["unsupported_addition_error"].fillna("").astype(str).str.strip().ne("")]
elif filter_option == "Mismatch explanation errors":
    view = view[
        view["empathy_mismatch_error"].fillna("").astype(str).str.strip().ne("")
        | view["quality_mismatch_error"].fillna("").astype(str).str.strip().ne("")
    ]

st.caption(f"Showing {len(view)} rows after the selected filter.")

summary_cols = [c for c in [
    "id", "general_category", "category", "question_word_count", "answer_word_count",
    "response_order", "gold_choice",
    "empathy_pred_source", "quality_pred_source", "factual_preservation_answer", "unsupported_addition_answer",
    "empathy_mismatch_explanation", "quality_mismatch_explanation",
    "generation_error", "empathy_judge_error", "quality_judge_error", "empathy_mismatch_error", "quality_mismatch_error",
] if c in view.columns]

if view.empty:
    st.info("No rows match this filter. Change the filter or run/generate/judge more rows.")
else:
    display_default = min(200, len(view))
    max_display_rows = st.number_input(
        "Maximum rows to display",
        min_value=1,
        max_value=max(1, len(view)),
        value=max(1, display_default),
        step=1 if len(view) < 50 else 50,
    )
    st.dataframe(view[summary_cols].head(int(max_display_rows)), use_container_width=True, height=360)

    st.markdown("### Inspect one row")
    inspect_ids = view["id"].astype(str).tolist()
    inspect_id = st.selectbox("Choose ID", inspect_ids, key=f"imb_inspect_{subset_key}_{slugify_label(gen_model_label)}_{slugify_label(judge_model_label)}_{filter_option}")
    row = view[view["id"].astype(str) == str(inspect_id)].iloc[0]

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("ID", normalize_text(row.get("id", "")))
    m2.metric("Category", normalize_text(row.get("general_category", ""))[:28])
    m3.metric("Order", normalize_text(row.get("response_order", "")))
    m4.metric("Empathy pref", normalize_text(row.get("empathy_pred_source", "")) or "—")
    m5.metric("Quality pref", normalize_text(row.get("quality_pred_source", "")) or "—")
    fp = normalize_text(row.get("factual_preservation_answer", ""))
    ua = normalize_text(row.get("unsupported_addition_answer", ""))
    fact_status = "Pass" if fp == "Yes" and ua == "No" else "Fail" if fp in ["Yes", "No"] and ua in ["Yes", "No"] else "—"
    m6.metric("Factual", fact_status)

    sent_a = get_sentence(row, "A")
    sent_b = get_sentence(row, "B")

    tab_texts, tab_judges, tab_prompts, tab_errors = st.tabs(["Texts", "Judge results", "Exact prompts", "Errors / metadata"])

    with tab_texts:
        st.markdown("**Patient question**")
        st.write(normalize_text(row.get("question", "")))
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("**Original answer**")
            st.write(normalize_text(row.get("original_answer", "")))
        with c_right:
            st.markdown("**Generated empathy-augmented answer**")
            st.write(normalize_text(row.get("generated_empathy_answer", "")))
        st.markdown("**A/B shown to judge**")
        ab1, ab2 = st.columns(2)
        with ab1:
            st.markdown(f"**A = {normalize_text(row.get('sentence_a_source', ''))}**")
            st.write(sent_a)
        with ab2:
            st.markdown(f"**B = {normalize_text(row.get('sentence_b_source', ''))}**")
            st.write(sent_b)

    with tab_judges:
        j1, j2, j3 = st.columns(3)
        with j1:
            st.markdown("#### Empathy")
            st.json({
                "pred_choice": normalize_text(row.get("empathy_pred_choice", "")),
                "pred_source": normalize_text(row.get("empathy_pred_source", "")),
                "judge_model": normalize_text(row.get("empathy_judge_model", "")),
                "judged_at": normalize_text(row.get("empathy_judged_at", "")),
            })
            st.markdown("**Raw judge response**")
            st.code(normalize_text(row.get("empathy_raw_judge_response", "")) or "", language="text")
            if normalize_text(row.get("empathy_mismatch_explanation", "")):
                st.markdown("**Why generated was not preferred**")
                st.write(normalize_text(row.get("empathy_mismatch_explanation", "")))
        with j2:
            st.markdown("#### Quality")
            st.json({
                "pred_choice": normalize_text(row.get("quality_pred_choice", "")),
                "pred_source": normalize_text(row.get("quality_pred_source", "")),
                "judge_model": normalize_text(row.get("quality_judge_model", "")),
                "judged_at": normalize_text(row.get("quality_judged_at", "")),
            })
            st.markdown("**Raw judge response**")
            st.code(normalize_text(row.get("quality_raw_judge_response", "")) or "", language="text")
            if normalize_text(row.get("quality_mismatch_explanation", "")):
                st.markdown("**Why generated was not preferred**")
                st.write(normalize_text(row.get("quality_mismatch_explanation", "")))
        with j3:
            st.markdown("#### Factual checks")
            fp = normalize_text(row.get("factual_preservation_answer", ""))
            ua = normalize_text(row.get("unsupported_addition_answer", ""))
            fact_status = "Pass" if fp == "Yes" and ua == "No" else "Fail" if fp in ["Yes", "No"] and ua in ["Yes", "No"] else "—"
            st.json({
                "factual_preservation_answer": fp,
                "unsupported_addition_answer": ua,
                "factual_status": fact_status,
                "factual_preservation_judge_model": normalize_text(row.get("factual_preservation_judge_model", "")),
                "unsupported_addition_judge_model": normalize_text(row.get("unsupported_addition_judge_model", "")),
            })
            st.markdown("**Preservation raw response**")
            st.code(normalize_text(row.get("factual_preservation_raw_response", "")) or "", language="text")
            st.markdown("**Unsupported-addition raw response**")
            st.code(normalize_text(row.get("unsupported_addition_raw_response", "")) or "", language="text")

    with tab_prompts:
        st.markdown("#### Generation prompt for this row")
        sys_g, user_g = generation_prompts(generation_prompt_variant_key)
        st.markdown("**System**")
        st.code(sys_g, language="text")
        st.markdown("**User**")
        st.code(user_g.format(question=normalize_text(row.get("question", "")), original_answer=normalize_text(row.get("original_answer", ""))), language="text")

        st.markdown("#### Empathy judge prompt for this row")
        sys_e, _ = judge_prompts("empathy")
        st.markdown("**System**")
        st.code(sys_e, language="text")
        st.markdown("**User**")
        st.code(exact_judge_user_prompt(row, "empathy"), language="text")

        st.markdown("#### Quality judge prompt for this row")
        sys_q, _ = judge_prompts("quality")
        st.markdown("**System**")
        st.code(sys_q, language="text")
        st.markdown("**User**")
        st.code(exact_judge_user_prompt(row, "quality"), language="text")

        st.markdown("#### Factual preservation check prompts for this row")
        sys_fp, _ = factual_check_prompts("preservation")
        st.markdown("**Preservation system**")
        st.code(sys_fp, language="text")
        st.markdown("**Preservation user**")
        st.code(exact_factual_user_prompt(row, "preservation"), language="text")
        sys_ua, _ = factual_check_prompts("unsupported_addition")
        st.markdown("**Unsupported-addition system**")
        st.code(sys_ua, language="text")
        st.markdown("**Unsupported-addition user**")
        st.code(exact_factual_user_prompt(row, "unsupported_addition"), language="text")

        if normalize_text(row.get("empathy_pred_source", "")) == "Original" or normalize_text(row.get("quality_pred_source", "")) == "Original":
            st.markdown("#### Mismatch explanation prompts")
            for dim in ["empathy", "quality"]:
                if normalize_text(row.get(f"{dim}_pred_source", "")) == "Original":
                    sys_m, user_m = mismatch_explanation_prompts(dim)
                    st.markdown(f"**{dim.title()} explanation system**")
                    st.code(sys_m, language="text")
                    st.markdown(f"**{dim.title()} explanation user**")
                    st.code(user_m.format(
                        question=normalize_text(row.get("question", "")),
                        original_answer=normalize_text(row.get("original_answer", "")),
                        generated_answer=normalize_text(row.get("generated_empathy_answer", "")),
                        raw_judge_response=normalize_text(row.get(f"{dim}_raw_judge_response", "")),
                    ), language="text")

    with tab_errors:
        meta = {
            "generation_model": normalize_text(row.get("generation_model", "")),
            "generated_at": normalize_text(row.get("generated_at", "")),
            "generation_error": normalize_text(row.get("generation_error", "")),
            "empathy_judge_error": normalize_text(row.get("empathy_judge_error", "")),
            "quality_judge_error": normalize_text(row.get("quality_judge_error", "")),
            "factual_preservation_error": normalize_text(row.get("factual_preservation_error", "")),
            "unsupported_addition_error": normalize_text(row.get("unsupported_addition_error", "")),
            "empathy_mismatch_error": normalize_text(row.get("empathy_mismatch_error", "")),
            "quality_mismatch_error": normalize_text(row.get("quality_mismatch_error", "")),
            "url": normalize_text(row.get("url", "")),
        }
        st.json({k: v for k, v in meta.items() if v})
