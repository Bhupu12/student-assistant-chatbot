# Student Personal Assistant Chatbot (Python)

A terminal-based chatbot that helps manage study tasks and answers basic NLP questions in English.

## Features
- Add / show / delete study tasks
- Parses due dates with regex (e.g., `2026-01-25` or `25 Jan`)
- Answers NLP questions using TF-IDF similarity matching (FAQ retrieval)

## Advanced NLP methods used
- Regex: date extraction and command parsing (`chatbot/nlp.py`)
- TF-IDF: question-to-answer matching (`FaqMatcher` in `chatbot/nlp.py`)

## How to run
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_chatbot.py
