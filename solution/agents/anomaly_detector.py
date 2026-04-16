"""
Agent 3 — Anomaly Detector
Model: google/gemini-2.5-flash  ($0.30 / $2.50 per 1M tokens)

Role:
    Cross-correlates three independent signal sources to make a per-transaction
    fraud assessment:

      Signal A — Statistical anomalies  (from StatisticalEngine)
      Signal B — Profiler findings       (from Agent 2)
      Signal C — Phishing events         (from Agent 1, time-correlated)

    Cross-correlation rules applied inside the prompt:
      • Phishing event ≤ 72 h before suspicious tx → strong multiplier
      • Statistical z-score > 3 AND new recipient → HIGH confidence
      • Description month mismatch alone → HIGH confidence
      • Single LOW signal → skip (too noisy)
"""
import json
import os
from datetime import datetime, timedelta

import pandas as pd
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
        temperature=0.1,
        max_tokens=2000,
    )


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def _filter_phishing_for_user(
    phishing_events: list,
    user_name: str,
    suspicious_tx_dates: list[str],
) -> list:
    """
    Keep only phishing events that target this user AND fall within
    72 hours before at least one suspicious transaction.
    """
    relevant = []
    for event in phishing_events:
        if user_name.lower() not in event.get("target_user", "").lower():
            continue
        try:
            event_dt = datetime.fromisoformat(event["date"])
        except (ValueError, KeyError):
            relevant.append(event)  # keep if we can't parse date
            continue
        for tx_date_str in suspicious_tx_dates:
            try:
                tx_dt = datetime.fromisoformat(tx_date_str[:19])
                if timedelta(0) <= (tx_dt - event_dt) <= timedelta(hours=72):
                    relevant.append(event)
                    break
            except ValueError:
                pass
    return relevant


@observe()
def run_anomaly_detector(
    session_id: str,
    user_id: str,
    user_name: str,
    user_profile: dict,
    transactions_df: pd.DataFrame,
    phishing_events: list,
    stat_scores: list,
) -> dict:
    """
    Cross-correlate all signals and classify each suspicious transaction.

    Returns
    -------
    {
      "fraud_transactions": [
          {
            "transaction_id": "...",
            "confidence": "HIGH | MEDIUM | LOW",
            "signals_fired": ["PHISHING_CORRELATION", "STAT_ANOMALY", ...],
            "explanation": "brief"
          }
      ]
    }
    """
    model = _get_model()
    langfuse_handler = CallbackHandler()

    # Gather candidate suspicious transactions from profiler + stat engine
    profiler_suspects = user_profile.get("suspicious_transactions", [])
    high_stat = [s for s in stat_scores if s.get("anomaly_score", 0) >= 20]

    # Collect dates of suspicious transactions for phishing time-window filter
    suspect_ids = {s["transaction_id"] for s in profiler_suspects + high_stat}
    suspect_dates = (
        transactions_df[transactions_df["transaction_id"].isin(suspect_ids)]["timestamp"]
        .astype(str)
        .tolist()
    )

    relevant_phishing = _filter_phishing_for_user(phishing_events, user_name, suspect_dates)

    # If no signals at all, skip expensive LLM call
    if not profiler_suspects and not high_stat and not relevant_phishing:
        return {"fraud_transactions": []}

    prompt = f"""You are a senior fraud detection specialist performing final anomaly analysis.

User: {user_id}  ({user_name})
Normal profile summary:
{json.dumps(user_profile.get("normal_profile", {}), indent=2)}

=== SIGNAL A — Statistical anomalies (pre-computed) ===
{json.dumps(high_stat, indent=2)}

=== SIGNAL B — Transaction profiler suspicious findings ===
{json.dumps(profiler_suspects, indent=2)}

=== SIGNAL C — Phishing / social-engineering events (≤72h before suspect tx) ===
{json.dumps(relevant_phishing, indent=2)}

CROSS-CORRELATION DECISION RULES (apply in this order):
1. SIGNAL_C fired ≤72h before a suspicious tx AND SIGNAL_B or SIGNAL_A also fire → FRAUD HIGH
2. SIGNAL_A: z-score > 3 AND new recipient (NEW_RECIPIENT flag) → FRAUD HIGH
3. SIGNAL_A: DESCRIPTION_MONTH_MISMATCH flag present → FRAUD HIGH
4. SIGNAL_A: VERY_EARLY_HOUR + NEW_RECIPIENT → FRAUD HIGH
5. Only SIGNAL_B at LOW confidence, no stat backing → LEGITIMATE (too noisy)
6. Prefer false-negative over false-positive when signals are weak.

For each transaction you classify as FRAUD, name ALL signals that fired.

Return ONLY valid JSON — no markdown, no extra text:
{{
  "fraud_transactions": [
    {{
      "transaction_id": "full UUID",
      "confidence": "HIGH | MEDIUM | LOW",
      "signals_fired": ["PHISHING_CORRELATION", "STAT_ANOMALY", "NEW_RECIPIENT", "DESCRIPTION_MISMATCH", "ODD_HOUR"],
      "explanation": "one sentence"
    }}
  ]
}}"""

    response = model.invoke(
        [HumanMessage(content=prompt)],
        config={
            "callbacks": [langfuse_handler],
            "metadata": {"langfuse_session_id": session_id},
        },
    )

    try:
        return json.loads(_strip_fences(response.content))
    except json.JSONDecodeError:
        return {"fraud_transactions": [], "_raw": response.content[:300]}
