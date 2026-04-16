"""
Agent 4 — Location Correlator
Model: google/gemini-2.0-flash-lite-001  ($0.075 / $0.30 per 1M tokens)
       (LLM only called when hard-to-interpret anomalies exist)

Role:
    For transactions that carry a physical location (in-person payments,
    ATM withdrawals) cross-check whether the user's GPS position at that
    time matches the transaction location.

    Pure-Python haversine distance does the heavy lifting; the LLM is
    invoked only when distance > threshold to write a human-readable
    explanation of the mismatch.
"""
import json
import math
import os

import pandas as pd
from langfuse import observe
from langfuse.langchain import CallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

MODEL_ID = "google/gemini-2.0-flash-lite-001"
DISTANCE_THRESHOLD_KM = 50  # flag if user GPS > 50 km from home city during tx


# ------------------------------------------------------------------ #
#  Pure computation helpers (no LLM)                                  #
# ------------------------------------------------------------------ #

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def compute_location_anomalies(
    transactions_df: pd.DataFrame,
    locations_df: pd.DataFrame,
    users: list,
) -> list[dict]:
    """
    For every transaction that has a location string, look up the user's
    nearest GPS fix within ±2 hours and compute the distance from home.

    Returns a list of anomaly dicts (only for transactions where we have
    sufficient data to make a judgement).
    """
    results: list[dict] = []

    loc_txs = transactions_df[transactions_df["location"].notna()].copy()
    if loc_txs.empty:
        return results

    locations_df = locations_df.copy()
    locations_df["timestamp"] = pd.to_datetime(locations_df["timestamp"])

    # Build IBAN → user mapping
    iban_to_user = {u["iban"]: u for u in users}

    for _, tx in loc_txs.iterrows():
        user_id = str(tx["sender_id"])
        sender_iban = str(tx.get("sender_iban", ""))
        tx_time = pd.to_datetime(tx["timestamp"])

        # Look up user info via IBAN
        user_info = iban_to_user.get(sender_iban, {})
        if not user_info:
            continue

        residence = user_info.get("residence", {})
        try:
            home_lat = float(residence.get("lat", 0))
            home_lon = float(residence.get("lng", 0))
        except (ValueError, TypeError):
            continue

        # Find closest GPS fix within ±2 hours
        user_gps = locations_df[locations_df["biotag"] == user_id].copy()
        if user_gps.empty:
            continue

        user_gps["time_diff_h"] = (
            (user_gps["timestamp"] - tx_time).abs().dt.total_seconds() / 3600
        )
        nearest = user_gps[user_gps["time_diff_h"] <= 2].nsmallest(1, "time_diff_h")
        if nearest.empty:
            continue

        gps_lat = float(nearest.iloc[0]["lat"])
        gps_lon = float(nearest.iloc[0]["lng"])
        gps_city = str(nearest.iloc[0].get("city", "unknown"))
        dist_home = _haversine(gps_lat, gps_lon, home_lat, home_lon)

        results.append(
            {
                "transaction_id": tx["transaction_id"],
                "tx_location": tx["location"],
                "user_gps_city": gps_city,
                "user_gps_coords": f"{gps_lat:.4f},{gps_lon:.4f}",
                "home_city": residence.get("city", "unknown"),
                "distance_from_home_km": round(dist_home, 1),
                "location_anomaly": dist_home > DISTANCE_THRESHOLD_KM,
            }
        )

    return results


# ------------------------------------------------------------------ #
#  LLM Agent — called only when anomalies exist                        #
# ------------------------------------------------------------------ #

def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


@observe()
def run_location_correlator(session_id: str, location_anomalies: list) -> dict:
    """
    Interprets computed location anomalies.
    Skips the LLM call when there are no anomalies (saves tokens).

    Returns
    -------
    {
      "location_fraud": [
          {"transaction_id": "...", "reason": "...", "confidence": "HIGH"}
      ]
    }
    """
    hard_anomalies = [a for a in location_anomalies if a.get("location_anomaly")]

    if not hard_anomalies:
        return {"location_fraud": [], "note": "No significant location mismatches found."}

    model = ChatOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model=MODEL_ID,
        temperature=0.1,
        max_tokens=800,
    )
    langfuse_handler = CallbackHandler()

    prompt = f"""You are a fraud analyst checking GPS vs transaction location mismatches.

Location anomalies detected (user GPS far from home during transaction):
{json.dumps(hard_anomalies, indent=2)}

A large distance between the user's GPS position and their home city at the time
of a transaction suggests either a compromised account or card fraud.

Return ONLY valid JSON — no markdown, no extra text:
{{
  "location_fraud": [
    {{
      "transaction_id": "full UUID",
      "reason": "one sentence",
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
        return {"location_fraud": []}
