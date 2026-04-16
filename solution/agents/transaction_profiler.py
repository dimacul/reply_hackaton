"""
Agent 2 — Transaction Profiler
Model: deepseek/deepseek-chat  ($0.32 / $0.89 per 1M tokens)

Role:
    Builds a per-user behavioral baseline from that user's own transaction
    history (cold-start safe: no cross-user data needed).

    It receives:
      - The user's full transaction list (already sorted chronologically)
      - Pre-computed statistical anomaly scores from StatisticalEngine
      - User demographic info (job, salary, city)

    It returns a structured profile + a list of transactions the LLM
    considers suspicious, with confidence levels.

    Cold-start strategy:
      The first 4-6 transactions are treated as the "normal window".
      Anything that deviates from that window raises a flag.
"""
import json
import os

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


@observe()
def run_transaction_profiler(
    session_id: str,
    user_id: str,
    user_info: dict,
    transactions_df: pd.DataFrame,
    stat_scores: list,
) -> dict:
    """
    Build a behavioral profile for *user_id* and flag suspicious transactions.

    Returns
    -------
    {
      "user_id": "...",
      "normal_profile": {
          "salary_source": "EMP_xxx or null",
          "salary_range": [min, max],
          "known_recipients": {"RECIPIENT_ID": "purpose"},
          "typical_hours": [8, 9, ...],
          "typical_amount_range": [min, max]
      },
      "suspicious_transactions": [
          {"transaction_id": "...", "reason": "...", "confidence": "HIGH"}
      ]
    }
    """
    model = _get_model()
    langfuse_handler = CallbackHandler()

    user_txs = (
        transactions_df[transactions_df["sender_id"] == user_id]
        .sort_values("timestamp")
        .copy()
    )

    # Build compact transaction lines, annotated with pre-computed stat scores
    stat_map = {s["transaction_id"]: s for s in stat_scores}
    tx_lines = []
    for _, tx in user_txs.iterrows():
        stat = stat_map.get(tx["transaction_id"], {})
        score_tag = ""
        if stat.get("anomaly_score", 0) > 0:
            score_tag = f" [STAT_SCORE={stat['anomaly_score']}, FLAGS={stat.get('flags', [])}]"
        tx_lines.append(
            f"{tx['timestamp']} | {tx['transaction_type']} | €{tx['amount']:.2f}"
            f" → {tx['recipient_id']} | desc={tx.get('description', '')} "
            f"| bal_after={tx['balance_after']}{score_tag}"
        )

    # High-score items to highlight
    high_stat = [s for s in stat_scores if s.get("anomaly_score", 0) >= 20]

    name = f"{user_info.get('first_name', '')} {user_info.get('last_name', '')}".strip()
    prompt = f"""You are a financial fraud analyst building a behavioral profile for a bank customer.

Customer: {name or user_id}
Job: {user_info.get("job", "unknown")} | Annual salary: {user_info.get("salary", "unknown")}
Home city: {user_info.get("residence", {}).get("city", "unknown")}

=== Full transaction history (chronological) ===
{chr(10).join(tx_lines)}

=== Statistical pre-analysis (high-score transactions) ===
{json.dumps(high_stat, indent=2)}

Instructions:
1. Identify the NORMAL behavioral pattern: regular salary receipts, regular rent/bills,
   typical shopping behaviour.
2. List KNOWN LEGITIMATE recipients with their typical payment purpose and amount range.
3. Flag any transaction that deviates from the normal pattern. Pay special attention to:
   - New or unknown recipient IDs (not seen in the first months)
   - Description says wrong month (e.g. "April" payment made in June)
   - Transaction at an unusual hour (before 06:00 or after 23:00)
   - Amount that is a statistical outlier vs. payments to the same recipient
   - A new payment category that never appeared before (e.g. "gym fee")

Return ONLY valid JSON — no markdown, no explanation outside the JSON:
{{
  "user_id": "{user_id}",
  "normal_profile": {{
    "salary_source": "EMP_xxx or null",
    "salary_range": [min, max],
    "known_recipients": {{"RECIPIENT_ID": "purpose and typical amount"}},
    "typical_hours": [list of hours seen in normal transactions],
    "typical_amount_range": [min, max]
  }},
  "suspicious_transactions": [
    {{
      "transaction_id": "full UUID",
      "reason": "specific reason",
      "confidence": "HIGH | MEDIUM | LOW"
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
        return {
            "user_id": user_id,
            "normal_profile": {},
            "suspicious_transactions": [],
            "_raw": response.content[:300],
        }
