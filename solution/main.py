"""
Reply Mirror — Fraud Detection System
Level 1: The Truman Show

Entry point. Follows the template from how-to-track-your-submission/main.py:
  - LangChain + OpenRouter for all LLM calls
  - Langfuse @observe() tracing on every agent call
  - session_id in format {TEAM_NAME}-{ULID} passed via metadata

Usage:
    python main.py

    Env vars (set in .env):
        OPENROUTER_API_KEY   — your OpenRouter key
        LANGFUSE_PUBLIC_KEY  — from the challenge dashboard
        LANGFUSE_SECRET_KEY  — from the challenge dashboard
        LANGFUSE_HOST        — https://challenges.reply.com/langfuse  (default)
        TEAM_NAME            — your team name (used in session_id)
        TRAIN_DIR            — path to training data folder
        VAL_DIR              — path to validation data folder
        OUTPUT_FILE          — where to write the submission (default: output.txt)
"""
import json
import os
from pathlib import Path

import pandas as pd
import ulid
from dotenv import load_dotenv
from langfuse import Langfuse

from agents.communication_scanner import run_communication_scan
from agents.transaction_profiler import run_transaction_profiler
from agents.anomaly_detector import run_anomaly_detector
from agents.location_correlator import compute_location_anomalies, run_location_correlator
from agents.orchestrator import run_orchestrator
from statistical_engine import StatisticalEngine

load_dotenv()

# ── Langfuse client (for final flush) ───────────────────────────────────────
langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse"),
)


def generate_session_id() -> str:
    team = os.getenv("TEAM_NAME", "team").replace(" ", "-")
    return f"{team}-{ulid.new().str}"


# ── Data loading ─────────────────────────────────────────────────────────────

def load_dataset(data_dir: str) -> dict:
    base = Path(data_dir)

    with open(base / "users.json", encoding="utf-8") as f:
        users = json.load(f)
    with open(base / "sms.json", encoding="utf-8") as f:
        sms = json.load(f)
    with open(base / "mails.json", encoding="utf-8") as f:
        mails = json.load(f)
    with open(base / "locations.json", encoding="utf-8") as f:
        locations = json.load(f)

    transactions = pd.read_csv(base / "transactions.csv")
    transactions["timestamp"] = pd.to_datetime(transactions["timestamp"])

    locations_df = pd.DataFrame(locations)

    return {
        "users": users,
        "sms": sms,
        "mails": mails,
        "locations": locations,
        "locations_df": locations_df,
        "transactions": transactions,
    }


# ── User ID ↔ user info resolution ──────────────────────────────────────────

def build_user_lookup(transactions: pd.DataFrame, users: list) -> dict:
    """
    Returns { sender_id -> user_info_dict } by matching sender_iban.
    Handles the cold-start case where val users are absent from train.
    """
    iban_to_user = {u["iban"]: u for u in users}
    lookup: dict = {}

    for _, tx in transactions.iterrows():
        sid = str(tx["sender_id"])
        if sid.startswith("EMP") or sid in lookup:
            continue
        sender_iban = str(tx.get("sender_iban", ""))
        if sender_iban in iban_to_user:
            lookup[sid] = iban_to_user[sender_iban]

    return lookup


# ── Main pipeline ────────────────────────────────────────────────────────────

def main():
    TRAIN_DIR   = os.getenv("TRAIN_DIR",   "The Truman Show - train")
    VAL_DIR     = os.getenv("VAL_DIR",     "The Truman Show - validation")
    OUTPUT_FILE = os.getenv("OUTPUT_FILE", "output.txt")

    session_id = generate_session_id()
    print(f"{'='*60}")
    print(f"Session ID : {session_id}")
    print(f"Train dir  : {TRAIN_DIR}")
    print(f"Val dir    : {VAL_DIR}")
    print(f"{'='*60}\n")

    # ── Load ────────────────────────────────────────────────────────────────
    print("[1/7] Loading datasets...")
    train = load_dataset(TRAIN_DIR)
    val   = load_dataset(VAL_DIR)

    # ── Step 0: Statistical Engine ──────────────────────────────────────────
    print("[2/7] Statistical analysis...")

    # Learn meta-patterns from training data (global priors)
    train_engine = StatisticalEngine(train["transactions"])
    meta = train_engine.learn_meta_patterns()
    print(f"      Meta-patterns: {meta}")

    # Score every outgoing transaction per validation user
    val_engine  = StatisticalEngine(val["transactions"])
    val_user_lookup = build_user_lookup(val["transactions"], val["users"])
    val_user_ids = list(val_user_lookup.keys())

    stat_scores_per_user: dict[str, list] = {}
    for uid in val_user_ids:
        scores = val_engine.score_all_for_user(uid)
        stat_scores_per_user[uid] = scores
        high = [s for s in scores if s["anomaly_score"] >= 20]
        if high:
            print(f"      [{uid}] High-score txs: "
                  f"{[(s['transaction_id'][:8], s['anomaly_score'], s['flags']) for s in high]}")

    # ── Agent 1 : Communication Scanner ────────────────────────────────────
    print("\n[3/7] Agent 1 — Communication Scanner...")
    comm_result = run_communication_scan(
        session_id=session_id,
        sms_data=val["sms"],
        mail_data=val["mails"],
        users=val["users"],
    )
    n_phishing = len(comm_result.get("phishing_events", []))
    print(f"      Phishing events found : {n_phishing}")
    print(f"      Compromised users     : {comm_result.get('compromised_users', [])}")

    # ── Agents 2 + 3 : Profiler + Anomaly Detector (per user) ──────────────
    print("\n[4/7] Agents 2 & 3 — Transaction Profiler + Anomaly Detector...")
    profiler_results: list[dict] = []
    anomaly_results:  list[dict] = []

    for uid in val_user_ids:
        user_info = val_user_lookup.get(uid, {})
        user_name = f"{user_info.get('first_name', '')} {user_info.get('last_name', '')}".strip()
        stat_scores = stat_scores_per_user.get(uid, [])

        print(f"\n      → [{uid}]  ({user_name})")

        # Agent 2 — Transaction Profiler
        profiler_result = run_transaction_profiler(
            session_id=session_id,
            user_id=uid,
            user_info=user_info,
            transactions_df=val["transactions"],
            stat_scores=stat_scores,
        )
        profiler_results.append(profiler_result)
        suspects_2 = [t["transaction_id"][:8]
                      for t in profiler_result.get("suspicious_transactions", [])]
        print(f"        Profiler suspects : {suspects_2}")

        # Agent 3 — Anomaly Detector
        anomaly_result = run_anomaly_detector(
            session_id=session_id,
            user_id=uid,
            user_name=user_name,
            user_profile=profiler_result,
            transactions_df=val["transactions"],
            phishing_events=comm_result.get("phishing_events", []),
            stat_scores=stat_scores,
        )
        anomaly_results.append(anomaly_result)
        fraud_3 = [t["transaction_id"][:8]
                   for t in anomaly_result.get("fraud_transactions", [])]
        print(f"        Anomaly fraud IDs : {fraud_3}")

    # ── Agent 4 : Location Correlator ──────────────────────────────────────
    print("\n[5/7] Agent 4 — Location Correlator...")
    loc_anomalies = compute_location_anomalies(
        val["transactions"],
        val["locations_df"],
        val["users"],
    )
    print(f"      Location anomalies computed : {len(loc_anomalies)}")

    location_result = run_location_correlator(
        session_id=session_id,
        location_anomalies=loc_anomalies,
    )
    loc_fraud = [t["transaction_id"][:8] for t in location_result.get("location_fraud", [])]
    print(f"      Location fraud IDs : {loc_fraud}")

    # ── Agent 5 : Orchestrator ──────────────────────────────────────────────
    print("\n[6/7] Agent 5 — Orchestrator (final decision)...")
    all_tx_ids = val["transactions"]["transaction_id"].tolist()

    fraud_ids = run_orchestrator(
        session_id=session_id,
        all_transaction_ids=all_tx_ids,
        communication_result=comm_result,
        profiler_results=profiler_results,
        anomaly_results=anomaly_results,
        location_result=location_result,
    )

    # ── Output ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"FRAUD TRANSACTIONS IDENTIFIED ({len(fraud_ids)}):")
    for tid in fraud_ids:
        print(f"  {tid}")
    print(f"{'='*60}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        fout.write("\n".join(fraud_ids))
    print(f"\n[7/7] Output written → {OUTPUT_FILE}")

    # ── Flush Langfuse traces ───────────────────────────────────────────────
    langfuse_client.flush()
    print(f"\nAll traces flushed to Langfuse | session: {session_id}")
    print("Dashboard may take a few minutes to update.")


if __name__ == "__main__":
    main()
