import json
import re
from typing import Optional, Dict, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Intent detection (keywords + regex)
# -----------------------------

_INTENT_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("show_tasks", re.compile(r"\b(show|list|view)\b.*\btasks?\b", re.IGNORECASE)),
    ("add_task", re.compile(r"\b(add|create|new)\b.*\btask\b|^add\s+task\s*:", re.IGNORECASE)),
    ("delete_task", re.compile(r"\b(delete|remove)\b.*\btask\b", re.IGNORECASE)),
]

def detect_intent(text: str) -> str:
    t = text.strip().lower()

    if t in {"exit", "quit", "bye", "goodbye"}:
        return "exit"
    if t in {"help", "h", "?", "commands"}:
        return "help"

    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(text):
            return intent

    # Anything else: try FAQ matcher
    return "faq"


# -----------------------------
# Regex parsing (advanced NLP method)
# -----------------------------

# ISO date: 2026-01-25
DATE_ISO = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")

# Numeric date formats: 25/01/2026 or 25-01-2026
DATE_NUMERIC = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](20\d{2})\b")

# Day + month name: 25 Jan / 25 January
DATE_DMY = re.compile(
    r"\b(\d{1,2})\s*(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\b",
    re.IGNORECASE
)

def _extract_due_date(text: str) -> Optional[str]:
    """
    Extracts a due date in one of several formats and returns a simple string.
    We keep it as a string (not a datetime) to keep the project lightweight.
    """
    iso = DATE_ISO.search(text)
    if iso:
        return iso.group(1)

    num = DATE_NUMERIC.search(text)
    if num:
        day, month, year = num.group(1), num.group(2), num.group(3)
        # Return in ISO-like format for consistency (YYYY-MM-DD)
        return f"{year}-{int(month):02d}-{int(day):02d}"

    dmy = DATE_DMY.search(text)
    if dmy:
        day = dmy.group(1)
        month_word = dmy.group(2).title()
        return f"{day} {month_word}"

    return None


def parse_add_task(text: str) -> Optional[Dict[str, Optional[str]]]:
    """
    Supports inputs like:
      - add task: Finish Lab 6 by 25 Jan
      - add task Finish Lab 6 by 2026-01-25
      - add a task Finish Lab 6 by 25/01/2026
    Returns:
      {"text": "<task text>", "due": "<due string or None>"}
    """
    raw = text.strip()

    # Try to capture task content after "add task" (with or without colon)
    m = re.search(r"^\s*add\s+(?:a\s+)?task\s*:?\s*(.+)$", raw, re.IGNORECASE)
    if not m:
        # If user wrote "create task ..." or "new task ..." you can support that too
        m = re.search(r"^\s*(?:create|new)\s+task\s*:?\s*(.+)$", raw, re.IGNORECASE)
        if not m:
            return None

    body = m.group(1).strip()
    if not body:
        return None

    # Extract due date if user wrote "by ..."
    due = None
    by = re.search(r"\bby\b\s+(.+)$", body, re.IGNORECASE)
    if by:
        tail = by.group(1).strip()
        due = _extract_due_date(tail)

        # Remove "by ..." portion from the stored task text
        body = re.sub(r"\bby\b\s+.+$", "", body, flags=re.IGNORECASE).strip()

    if not body:
        return None

    return {"text": body, "due": due}


def parse_delete_task(text: str) -> Optional[int]:
    """
    Supports:
      - delete task 2
      - remove task 10
      - delete 3  (optional style)
    """
    m = re.search(r"\b(delete|remove)\s+task\s+(\d+)\b", text, re.IGNORECASE)
    if m:
        return int(m.group(2))

    # Optional convenience: "delete 2"
    m2 = re.search(r"^\s*(delete|remove)\s+(\d+)\s*$", text, re.IGNORECASE)
    if m2:
        return int(m2.group(2))

    return None


# -----------------------------
# TF-IDF FAQ matcher (advanced NLP method)
# -----------------------------

class FaqMatcher:
    def __init__(self, faq_path: str):
        with open(faq_path, "r", encoding="utf-8") as f:
            self.faq = json.load(f)

        self.questions = [item["q"] for item in self.faq]
        self.answers = [item["a"] for item in self.faq]

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.questions)

    def answer(self, user_text: str, threshold: float = 0.22) -> Optional[str]:
        """
        Returns best matching FAQ answer if similarity >= threshold.
        Threshold avoids random matches.
        """
        vec = self.vectorizer.transform([user_text])
        sims = cosine_similarity(vec, self.matrix)[0]

        best_i = int(sims.argmax())
        best_score = float(sims[best_i])

        if best_score < threshold:
            return None

        return self.answers[best_i]
