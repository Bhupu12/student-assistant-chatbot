import json
import re
from typing import Optional, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------- Intent detection (keywords + regex) ---------

def detect_intent(text: str) -> str:
    t = text.lower().strip()

    if t in {"exit", "quit", "bye"}:
        return "exit"
    if t in {"help", "h", "?"}:
        return "help"

    if re.search(r"\bshow\b.*\btasks\b|\blist\b.*\btasks\b", t):
        return "show_tasks"
    if re.search(r"^add\s+task\s*:", t) or re.search(r"\badd\b.*\btask\b", t):
        return "add_task"
    if re.search(r"\bdelete\b.*\btask\b|\bremove\b.*\btask\b", t):
        return "delete_task"

    # Anything else: try FAQ matcher
    return "faq"

# --------- Regex parsing (advanced NLP method) ---------

DATE_ISO = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")  # 2026-01-25
DATE_DMY = re.compile(
    r"\b(\d{1,2})\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b",
    re.IGNORECASE
)

def parse_add_task(text: str) -> Optional[Dict[str, str]]:
    """
    Accepts:
      add task: do something by 2026-01-25
      add task: do something by 25 Jan
    Returns dict with text + optional due string (kept simple for assessment).
    """
    m = re.search(r"add\s+task\s*:\s*(.+)", text, re.IGNORECASE)
    if not m:
        return None

    body = m.group(1).strip()

    # Extract due date if user wrote "by ..."
    due = None
    by = re.search(r"\bby\b\s+(.+)$", body, re.IGNORECASE)
    if by:
        tail = by.group(1).strip()

        iso = DATE_ISO.search(tail)
        if iso:
            due = iso.group(1)
        else:
            dmy = DATE_DMY.search(tail)
            if dmy:
                day = dmy.group(1)
                mon = dmy.group(2).title()
                due = f"{day} {mon}"

        # Remove "by ..." from task text for cleanliness
        body = re.sub(r"\bby\b\s+.+$", "", body, flags=re.IGNORECASE).strip()

    if not body:
        return None
    return {"text": body, "due": due}

def parse_delete_task(text: str) -> Optional[int]:
    m = re.search(r"\b(delete|remove)\s+task\s+(\d+)\b", text, re.IGNORECASE)
    if not m:
        return None
    return int(m.group(2))

# --------- TF-IDF FAQ matcher (advanced NLP method) ---------

class FaqMatcher:
    def __init__(self, faq_path: str):
        with open(faq_path, "r", encoding="utf-8") as f:
            self.faq = json.load(f)

        self.questions = [item["q"] for item in self.faq]
        self.answers = [item["a"] for item in self.faq]

        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.questions)

    def answer(self, user_text: str) -> Optional[str]:
        vec = self.vectorizer.transform([user_text])
        sims = cosine_similarity(vec, self.matrix)[0]
        best_i = int(sims.argmax())
        best_score = float(sims[best_i])

        # Threshold prevents random matches
        if best_score < 0.22:
            return None

        return self.answers[best_i]
