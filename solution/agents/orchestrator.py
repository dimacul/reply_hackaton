"""
Agent 5 — Orchestrator
Model: anthropic/claude-3.5-haiku  ($0.80 / $4.00 per 1M tokens)

Role:
    Receives the outputs of all four specialist agents, de-duplicates and
    weights their signals, and makes the FINAL fraud/legitimate decision
    for each transaction.

    Decision logic (applied inside the prompt):
      CONFIRM fraud  if ≥ 2 agents flag the same tx with HIGH / MEDIUM
      CONFIRM fraud  if Anomaly Detector alone reports HIGH confidence
      CONFIRM fraud  if phishing event correlated (PHISHING_CORRELATION signal)
      REJECT         if only 1 LOW confidence signal fires
      DEFAULT        prefer false-negative over false-positive

    Output: plain list of fraudulent transaction IDs, ready for submission.
"""
import json
import os
import re

from langfuse import observe
from langfuse.langchain import CallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

MODEL_ID = "anthropic/claude-sonnet-4.5"


def _get_model() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model=MODEL_ID,
        temperature=0.0,   # determinism wanted at final decision stage
        max_tokens=1000,
    )


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def _merge_signals(
    anomaly_results: list[dict],
    profiler_results: list[dict],
    location_result: dict,
) -> dict:
    """
    Build a unified signal map:
      { transaction_id: { sources: [...], signals: [...], max_confidence: "..." } }
    """
    merged: dict = {}

    def _add(tid: str, source: str, confidence: str, signals: list = None):
        if tid not in merged:
            merged[tid] = {"sources": [], "signals": [], "max_confidence": "LOW"}
        merged[tid]["sources"].append(source)
        if signals:
            merged[tid]["signals"].extend(signals)
        # Escalate confidence
        order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        if order.get(confidence, 0) > order.get(merged[tid]["max_confidence"], 0):
            merged[tid]["max_confidence"] = confidence

    for result in anomaly_results:
        for tx in result.get("fraud_transactions", []):
            _add(
                tx["transaction_id"],
                "anomaly_detector",
                tx.get("confidence", "LOW"),
                tx.get("signals_fired", []),
            )

    for result in profiler_results:
        for tx in result.get("suspicious_transactions", []):
            _add(
                tx["transaction_id"],
                "profiler",
                tx.get("confidence", "LOW"),
            )

    for tx in location_result.get("location_fraud", []):
        _add(
            tx["transaction_id"],
            "location_correlator",
            tx.get("confidence", "LOW"),
            ["LOCATION_MISMATCH"],
        )

    return merged


@observe()
def run_orchestrator(
    session_id: str,
    all_transaction_ids: list,
    communication_result: dict,
    profiler_results: list,
    anomaly_results: list,
    location_result: dict,
) -> list[str]:
    """
    Final decision agent.  Returns a list of fraudulent transaction IDs.
    """
    model = _get_model()
    langfuse_handler = CallbackHandler()

    merged = _merge_signals(anomaly_results, profiler_results, location_result)

    # Pre-filter: send all flagged transactions to the LLM
    # Include LOW confidence too — orchestrator will make the final call
    candidates = {
        tid: info for tid, info in merged.items()
        if info["max_confidence"] in ("HIGH", "MEDIUM", "LOW") or len(info["sources"]) >= 1
    }

    n_total = len(all_transaction_ids)

    prompt = f"""You are the final decision agent in a multi-agent fraud detection system.

Total transactions in dataset: {n_total}
Flagged by sub-agents ({len(candidates)} candidates):
{json.dumps(candidates, indent=2)}

Communication analysis:
- Compromised users: {communication_result.get("compromised_users", [])}
- Summary: {communication_result.get("summary", "")}

FINAL DECISION RULES:
1. FRAUD — confirmed if flagged by ≥ 2 independent agents (any confidence)
2. FRAUD — confirmed if anomaly_detector reports HIGH confidence
3. FRAUD — confirmed if "PHISHING_CORRELATION" in signals_fired (social engineering)
4. FRAUD — confirmed if "DESCRIPTION_MONTH_MISMATCH" present
5. FRAUD — confirmed if "VERY_EARLY_HOUR" + transaction at unusual time (before 06:00)
6. FRAUD — confirmed if "ONE_TIME_RECIPIENT" or "NEW_RECIPIENT" + high amount relative to description
   (e.g. an "internet bill" of €300+ is suspicious — typical internet bills are €20-50)
7. NOT FRAUD — regular salary receipts, recurring rent to known recipients, small daily purchases
8. NOT FRAUD — direct debits to recipients that appear multiple times (banks process at night legitimately)

Asymmetric cost reminder:
  Blocking a LEGITIMATE transaction = economic + reputational damage to the bank.
  Letting a FRAUDULENT transaction pass = financial loss.
  → Only flag transactions where evidence is clear.

The output list MUST contain between 1 and {n_total - 1} transaction IDs (never 0, never all).

Return ONLY a JSON array of confirmed fraudulent transaction IDs — full UUIDs, no truncation:
["uuid-1", "uuid-2"]"""

    response = model.invoke(
        [HumanMessage(content=prompt)],
        config={
            "callbacks": [langfuse_handler],
            "metadata": {"langfuse_session_id": session_id},
        },
    )

    raw = _strip_fences(response.content)

    valid = set(all_transaction_ids)

    try:
        result = json.loads(raw)
        if isinstance(result, list):
            # Deduplicate while preserving order, only keep valid IDs
            return list(dict.fromkeys(str(tid) for tid in result if tid in valid))
    except json.JSONDecodeError:
        pass

    # Fallback: extract UUIDs from raw text, deduplicate
    uuid_pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
    found = re.findall(uuid_pattern, raw, re.IGNORECASE)
    return list(dict.fromkeys(tid for tid in found if tid in valid))
