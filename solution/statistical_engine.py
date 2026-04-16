"""
Statistical Engine — pure Python/numpy, no LLM calls.
Builds behavioral profiles and computes anomaly scores mathematically.
Used as a fast pre-filter before the LLM agents.
"""
import numpy as np
import pandas as pd
from typing import Optional


class StatisticalEngine:

    def __init__(self, transactions_df: pd.DataFrame):
        self.df = transactions_df.copy()
        self.df["timestamp"] = pd.to_datetime(self.df["timestamp"])
        self.df["hour"] = self.df["timestamp"].dt.hour
        self.df["month"] = self.df["timestamp"].dt.month
        self.profiles: dict = {}

    # ------------------------------------------------------------------ #
    #  Profile building                                                    #
    # ------------------------------------------------------------------ #

    def build_profile(self, user_id: str) -> dict:
        """
        Build a behavioral profile from a user's outgoing transaction history.
        Handles cold-start: if < 3 txs exist, returns a minimal profile.
        """
        outgoing = self.df[self.df["sender_id"] == user_id].copy()
        if outgoing.empty:
            return {}

        # Per-recipient stats
        recipient_profiles: dict = {}
        for recipient, group in outgoing.groupby("recipient_id"):
            if pd.isna(recipient):
                continue
            amounts = group["amount"].values
            recipient_profiles[str(recipient)] = {
                "count": int(len(amounts)),
                "mean": float(np.mean(amounts)),
                "std": float(np.std(amounts)) if len(amounts) > 1 else 0.0,
                "min": float(np.min(amounts)),
                "max": float(np.max(amounts)),
                "usual_hours": group["hour"].tolist(),
                "usual_months": group["month"].tolist(),
                "types": group["transaction_type"].dropna().unique().tolist(),
            }

        hours = outgoing["hour"].values
        amounts = outgoing["amount"].values

        profile = {
            "user_id": user_id,
            "known_recipients": list(recipient_profiles.keys()),
            "recipient_profiles": recipient_profiles,
            "global_amount_mean": float(np.mean(amounts)),
            "global_amount_std": float(np.std(amounts)) if len(amounts) > 1 else 0.0,
            "active_hours_p5": float(np.percentile(hours, 5)) if len(hours) > 1 else 0.0,
            "active_hours_p95": float(np.percentile(hours, 95)) if len(hours) > 1 else 23.0,
            "transaction_types": outgoing["transaction_type"].dropna().unique().tolist(),
            "tx_count": int(len(outgoing)),
        }

        self.profiles[user_id] = profile
        return profile

    # ------------------------------------------------------------------ #
    #  Transaction scoring                                                 #
    # ------------------------------------------------------------------ #

    def score_transaction(self, tx: pd.Series, profile: dict) -> dict:
        """
        Compute an anomaly score [0-100] for a single outgoing transaction.
        Returns score + list of flag strings explaining the anomaly.
        """
        if not profile:
            return {"transaction_id": tx["transaction_id"], "anomaly_score": 0, "flags": []}

        flags: list[str] = []
        score: int = 0

        recipient = str(tx["recipient_id"]) if not pd.isna(tx.get("recipient_id")) else None
        amount = float(tx["amount"])
        hour = int(pd.to_datetime(tx["timestamp"]).hour)
        tx_type = str(tx.get("transaction_type", ""))
        description = str(tx.get("description", "")) if not pd.isna(tx.get("description")) else ""

        # 1. Unknown recipient (+50)
        known = profile.get("known_recipients", [])
        if recipient and recipient not in known:
            flags.append(f"NEW_RECIPIENT:{recipient}")
            score += 50

        # 2. Global amount z-score
        g_mean = profile.get("global_amount_mean", amount)
        g_std = profile.get("global_amount_std", 0.0)
        if g_std > 1.0:
            z_global = abs((amount - g_mean) / g_std)
            if z_global > 3:
                flags.append(f"AMOUNT_OUTLIER_GLOBAL:z={z_global:.1f}")
                score += min(int(z_global * 5), 40)

        # 3. Per-recipient amount z-score (strongest signal when we know the recipient)
        rp = profile.get("recipient_profiles", {}).get(recipient or "", {})
        if rp and rp.get("std", 0) > 1.0:
            z_recip = abs((amount - rp["mean"]) / rp["std"])
            if z_recip > 3:
                flags.append(f"AMOUNT_VS_RECIPIENT_OUTLIER:z={z_recip:.1f}")
                score += min(int(z_recip * 10), 50)

        # 4. Odd hour — absolute (before 06:00 or after 23:00)
        if hour < 6:
            flags.append(f"VERY_EARLY_HOUR:{hour:02d}:xx")
            score += 35
        elif hour >= 23:
            flags.append(f"VERY_LATE_HOUR:{hour:02d}:xx")
            score += 20

        # 5. Hour outside user's normal window
        p5 = profile.get("active_hours_p5", 0)
        p95 = profile.get("active_hours_p95", 23)
        if (hour < p5 or hour > p95) and "VERY_EARLY_HOUR" not in str(flags) and "VERY_LATE_HOUR" not in str(flags):
            flags.append(f"UNUSUAL_HOUR:{hour}h (normal {p5:.0f}h-{p95:.0f}h)")
            score += 15

        # 6. New transaction type
        if tx_type and tx_type not in profile.get("transaction_types", []):
            flags.append(f"NEW_TX_TYPE:{tx_type}")
            score += 15

        # 7. Description mismatch — month label vs actual month
        tx_month = pd.to_datetime(tx["timestamp"]).month
        month_names = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
        }
        for name, num in month_names.items():
            if name in description.lower() and num != tx_month:
                flags.append(f"DESCRIPTION_MONTH_MISMATCH:says_{name.upper()}_but_is_month_{tx_month}")
                score += 45
                break

        return {
            "transaction_id": tx["transaction_id"],
            "anomaly_score": min(score, 100),
            "flags": flags,
        }

    # ------------------------------------------------------------------ #
    #  Meta-pattern learning (from training data)                          #
    # ------------------------------------------------------------------ #

    def learn_meta_patterns(self) -> dict:
        """
        Extract meta-patterns from training data.
        These act as priors when we encounter entirely new users in validation.
        """
        all_hours: list[int] = []
        all_recipient_counts: list[int] = []
        all_tx_per_month: list[float] = []

        citizen_ids = [
            uid for uid in self.df["sender_id"].dropna().unique()
            if not str(uid).startswith("EMP")
        ]

        for uid in citizen_ids:
            profile = self.build_profile(uid)
            if not profile:
                continue
            outgoing = self.df[self.df["sender_id"] == uid]
            all_hours.extend(outgoing["hour"].tolist())
            all_recipient_counts.append(len(profile.get("known_recipients", [])))
            n_months = outgoing["month"].nunique()
            if n_months > 0:
                all_tx_per_month.append(len(outgoing) / n_months)

        meta: dict = {}
        if all_hours:
            meta["global_p5_hour"] = float(np.percentile(all_hours, 5))
            meta["global_p95_hour"] = float(np.percentile(all_hours, 95))
        if all_recipient_counts:
            meta["avg_recipients"] = float(np.mean(all_recipient_counts))
            meta["max_recipients"] = int(np.max(all_recipient_counts))
        if all_tx_per_month:
            meta["avg_tx_per_month"] = float(np.mean(all_tx_per_month))

        return meta

    # ------------------------------------------------------------------ #
    #  Convenience: score all transactions for a user                      #
    # ------------------------------------------------------------------ #

    def score_all_for_user(self, user_id: str) -> list[dict]:
        profile = self.build_profile(user_id)
        user_txs = self.df[self.df["sender_id"] == user_id]
        return [self.score_transaction(row, profile) for _, row in user_txs.iterrows()]
