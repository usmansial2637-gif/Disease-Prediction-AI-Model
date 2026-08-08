"""
chatbot.py
----------
A lightweight, self-contained symptom-checker chatbot (Level 10). It's
keyword/rule-based rather than calling an external LLM API, so the project
runs fully offline with no API key required — but it's structured so a real
LLM call could be dropped in later (see `respond()`).

It never claims a diagnosis. It maps mentioned symptoms to the dataset(s)
they're most associated with, and always points the user at the Predict
tool + a real doctor.
"""

import re

SYMPTOM_MAP = {
    "heart": {
        "label": "Heart Disease",
        "keywords": [
            "chest pain", "chest tightness", "shortness of breath", "breathless",
            "palpitations", "irregular heartbeat", "dizziness", "fainting",
            "high blood pressure", "hypertension", "heart", "arm pain", "jaw pain",
        ],
    },
    "diabetes": {
        "label": "Diabetes",
        "keywords": [
            "excessive thirst", "thirsty", "frequent urination", "peeing a lot",
            "blurred vision", "blurry vision", "weight loss", "hunger", "hungry",
            "numb feet", "tingling feet", "slow healing", "diabetes", "sugar",
        ],
    },
    "breast_cancer": {
        "label": "Breast Cancer",
        "keywords": [
            "breast lump", "lump in breast", "breast pain", "nipple discharge",
            "breast skin change", "dimpling", "breast", "lump",
        ],
    },
}

# Symptoms that don't map to one specific dataset but are still worth
# acknowledging so the bot doesn't look like it's ignoring the user.
GENERIC_SYMPTOMS = ["fatigue", "tired", "headache", "fever", "nausea", "weakness"]

GREETINGS = ["hi", "hello", "hey", "salam", "assalamualaikum"]


def _find_matches(message: str):
    msg = message.lower()
    matches = {}
    for dataset_key, info in SYMPTOM_MAP.items():
        hits = [kw for kw in info["keywords"] if kw in msg]
        if hits:
            matches[dataset_key] = hits
    generic_hits = [s for s in GENERIC_SYMPTOMS if s in msg]
    return matches, generic_hits


def respond(message: str) -> str:
    """
    Returns a plain-text chatbot reply. Rule-based today; swap the body of
    this function for a real LLM API call (OpenAI/Anthropic/etc.) later
    without touching the rest of the app — the calling code (app.py) only
    depends on this function's (str) -> str signature.
    """
    msg = message.strip()
    if not msg:
        return "I didn't catch that — could you describe what you're feeling?"

    msg_lower = msg.lower()
    if any(re.fullmatch(rf"{g}[!.\s]*", msg_lower) or msg_lower.startswith(g) for g in GREETINGS):
        return ("Hi! Tell me what symptoms you're experiencing (e.g. \"I have chest pain and "
                "shortness of breath\") and I'll point you toward the right prediction tool.")

    matches, generic_hits = _find_matches(msg)

    if not matches and not generic_hits:
        return ("I couldn't match that to any symptoms I recognize for heart disease, diabetes, "
                "or breast cancer screening. Could you describe your symptoms more specifically? "
                "For anything urgent, please contact a doctor directly rather than relying on this chatbot.")

    lines = []
    if matches:
        possibilities = ", ".join(SYMPTOM_MAP[k]["label"] for k in matches)
        if len(matches) == 1:
            lines.append(f"Based on your symptoms, {possibilities} is a possibility worth screening for.")
        else:
            lines.append(f"Based on your symptoms, {possibilities} are possibilities worth screening for.")
        for dataset_key, hits in matches.items():
            label = SYMPTOM_MAP[dataset_key]["label"]
            lines.append(f"  • {label} — matched: {', '.join(hits)}")
        lines.append("")
        lines.append("Please use the Predict tab (select the matching dataset above) for a "
                      "screening estimate, and consult a physician for an actual diagnosis.")
    else:
        lines.append(f"I noted: {', '.join(generic_hits)}. These are general symptoms that alone "
                      "don't point to a specific condition here — if they persist or worsen, "
                      "please see a doctor.")

    lines.append("")
    lines.append("⚠️ I'm not a substitute for medical advice. This is a screening aid only.")
    return "\n".join(lines)
