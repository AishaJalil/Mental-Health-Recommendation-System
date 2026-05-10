"""
MindMatch – Natural Language Condition Parser

Primary:  Google Gemini 1.5 Flash  (set GEMINI_API_KEY env variable)
Fallback: Keyword-based scoring    (works with no API key / offline)
"""

import os
import json
import re

try:
    import google.generativeai as genai
    _GEMINI_OK = True
except ImportError:
    _GEMINI_OK = False


# ---------------------------------------------------------------------------
# Gemini prompt
# ---------------------------------------------------------------------------

_PROMPT = """You are a mental health assessment assistant helping a wellness activity recommender system.

A user described how they are feeling in plain language. Your job is to estimate their depression, anxiety, and stress levels on the DASS-42 scale (0–42 each).

DASS-42 clinical severity bands:
- Depression: Normal 0–9 | Mild 10–13 | Moderate 14–20 | Severe 21–27 | Extremely Severe 28+
- Anxiety:    Normal 0–7  | Mild 8–9   | Moderate 10–14 | Severe 15–19 | Extremely Severe 20+
- Stress:     Normal 0–14 | Mild 15–18 | Moderate 19–25 | Severe 26–33 | Extremely Severe 34+

User's description:
"{text}"

Rules:
1. Be conservative — vague or mild phrasing should map to lower scores
2. If no symptoms are mentioned for a dimension, set it to 0
3. detected_symptoms must be short phrases taken directly from what the user said
4. explanation must be 1–2 plain-language sentences summarising what you detected

Respond ONLY with valid JSON, no markdown fences, no extra text:
{{
  "depression": <integer 0–42>,
  "anxiety": <integer 0–42>,
  "stress": <integer 0–42>,
  "detected_symptoms": ["...", "..."],
  "explanation": "..."
}}"""


# ---------------------------------------------------------------------------
# Keyword fallback (no API needed)
# ---------------------------------------------------------------------------

_KEYWORDS = {
    # Depression
    "hopeless":            {"depression": 9},
    "worthless":           {"depression": 9},
    "empty inside":        {"depression": 8},
    "no purpose":          {"depression": 8},
    "feel empty":          {"depression": 8},
    "no motivation":       {"depression": 7},
    "can't get out of bed":{"depression": 9},
    "no interest":         {"depression": 7},
    "don't enjoy":         {"depression": 7},
    "depressed":           {"depression": 10},
    "crying":              {"depression": 6},
    "lonely":              {"depression": 5},
    "pointless":           {"depression": 7},
    "giving up":           {"depression": 9},
    "sad":                 {"depression": 5},
    "low mood":            {"depression": 6},
    "numb":                {"depression": 7},
    "lost interest":       {"depression": 8},
    "feel down":           {"depression": 5},
    # Anxiety
    "anxious":             {"anxiety": 9},
    "anxiety":             {"anxiety": 10},
    "panic":               {"anxiety": 11},
    "panic attack":        {"anxiety": 14},
    "heart racing":        {"anxiety": 8},
    "can't breathe":       {"anxiety": 8},
    "shortness of breath": {"anxiety": 8},
    "sweating":            {"anxiety": 6},
    "worried":             {"anxiety": 7},
    "fear":                {"anxiety": 7},
    "nervous":             {"anxiety": 6},
    "dread":               {"anxiety": 8},
    "overthinking":        {"anxiety": 8},
    "racing thoughts":     {"anxiety": 8},
    "scared":              {"anxiety": 7},
    "trembling":           {"anxiety": 7},
    "on edge":             {"anxiety": 7},
    # Stress
    "stressed":            {"stress": 9},
    "overwhelmed":         {"stress": 9},
    "can't sleep":         {"stress": 8},
    "insomnia":            {"stress": 8},
    "exhausted":           {"stress": 6},
    "burnout":             {"stress": 10},
    "too much pressure":   {"stress": 8},
    "pressure":            {"stress": 6},
    "deadline":            {"stress": 5},
    "can't focus":         {"stress": 7},
    "can't concentrate":   {"stress": 7},
    "irritable":           {"stress": 6},
    "angry":               {"stress": 5},
    "snapping":            {"stress": 6},
    "tense":               {"stress": 6},
    "restless":            {"stress": 6, "anxiety": 4},
    "tired all the time":  {"stress": 6},
    "no energy":           {"stress": 5, "depression": 4},
}


def _keyword_fallback(text: str) -> dict:
    text_lower = text.lower()
    scores = {"depression": 0, "anxiety": 0, "stress": 0}
    detected = []

    for phrase, weights in _KEYWORDS.items():
        if phrase in text_lower:
            detected.append(phrase)
            for dim, val in weights.items():
                scores[dim] = min(42, scores[dim] + val)

    return {
        **scores,
        "detected_symptoms": detected,
        "explanation": (
            "Estimated from keyword analysis. "
            "Add a GEMINI_API_KEY environment variable for more accurate assessment."
        ),
        "source": "keyword_fallback",
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_condition(text: str) -> dict:
    """
    Parse a free-text condition description into DASS-42 scores.
    Uses Gemini if GEMINI_API_KEY is set, otherwise falls back to keywords.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    if not _GEMINI_OK or not api_key:
        return _keyword_fallback(text)

    # Models tried in order — first one that responds wins
    _MODELS = [
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
    ]

    try:
        genai.configure(api_key=api_key)
        prompt = _PROMPT.format(text=text.replace('"', "'"))

        raw = None
        last_err = None
        for model_name in _MODELS:
            try:
                model    = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                raw      = response.text.strip()
                print(f"  [Gemini] used model: {model_name}")
                break
            except Exception as e:
                last_err = e
                continue

        if raw is None:
            raise last_err

        # Strip markdown code fences if model adds them
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        result = json.loads(raw)

        # Clamp and validate
        for dim in ("depression", "anxiety", "stress"):
            result[dim] = max(0, min(42, int(result.get(dim, 0))))
        result.setdefault("detected_symptoms", [])
        result.setdefault("explanation", "")
        result["source"] = "gemini"
        return result

    except Exception as e:
        print(f"  [Gemini] parse_condition error: {e} — using keyword fallback")
        result         = _keyword_fallback(text)
        result["source"] = "keyword_fallback"
        return result
