"""
Agent 1 — Communication Scanner
Model: google/gemini-2.0-flash-lite-001  ($0.075 / $0.30 per 1M tokens)

Role:
    Scans SMS and email threads for phishing, social engineering, and
    suspicious URLs.  Outputs per-user phishing events with timestamps
    so the Anomaly Detector can correlate them with transactions.
"""
import json
import os

from langfuse import observe
from langfuse.langchain import CallbackHandler
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

MODEL_ID = "google/gemini-2.0-flash-lite-001"


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
def run_communication_scan(session_id: str, sms_data: list, mail_data: list, users: list) -> dict:
    """
    Scan all SMS and email communications for phishing / social-engineering signals.

    Returns
    -------
    {
      "phishing_events": [
          {
            "target_user": "First Last",
            "date": "YYYY-MM-DD",
            "channel": "sms" | "email",
            "description": "...",
            "fake_domain": "domain or null",
            "risk": "HIGH" | "MEDIUM" | "LOW"
          }
      ],
      "compromised_users": ["First Last"],
      "summary": "brief overall assessment"
    }
    """
    model = _get_model()
    langfuse_handler = CallbackHandler()

    user_names = [f"{u['first_name']} {u['last_name']}" for u in users]

    # Limit token usage: keep first 80 SMS, emails truncated to 400 chars each
    sms_text = "\n---\n".join(s["sms"] for s in sms_data[:80])
    mail_text = "\n---\n".join(m["mail"][:400] for m in mail_data)

    prompt = f"""You are a cybersecurity analyst. Detect phishing, social engineering, and suspicious communications.

Users in scope: {", ".join(user_names)}

=== SMS MESSAGES ===
{sms_text}

=== EMAIL EXCERPTS ===
{mail_text}

Phishing indicators to look for:
- Typosquatted domains (e.g. "ub3r-verify.com", "amaz0n-verify.com" — digits replacing letters)
- Urgent language: "Verify NOW", "account suspended", "action required"
- Unexpected requests for credentials or financial action
- Mismatched sender names vs. domains
- Messages that precede unusual financial transactions

For each suspicious message record the EXACT DATE from the message header.

Return ONLY valid JSON — no markdown, no explanation outside the JSON:
{{
  "phishing_events": [
    {{
      "target_user": "First Last",
      "date": "YYYY-MM-DD",
      "channel": "sms",
      "description": "brief description",
      "fake_domain": "domain or null",
      "risk": "HIGH"
    }}
  ],
  "compromised_users": ["First Last"],
  "summary": "one sentence assessment"
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
            "phishing_events": [],
            "compromised_users": [],
            "summary": response.content[:300],
        }
