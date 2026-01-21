# Student Personal Assistant Chatbot (Python)

A terminal-based chatbot written in Python that helps students manage study tasks and answer basic Natural Language Processing (NLP) questions through an English conversation interface.

## Features
- Add, view, and delete study tasks
- Parses due dates from natural language input using regex (e.g., `2026-01-25` or `25 Jan`)
- Answers NLP-related questions using TF-IDF similarity matching

## Advanced NLP Methods Used
- **Regex**: Used for command interpretation and date extraction from user input (implemented in `chatbot/nlp.py`)
- **TF-IDF**: Used for question-to-answer matching by comparing user queries with a small FAQ knowledge base using cosine similarity (`FaqMatcher` in `chatbot/nlp.py`)

## How to Run the Chatbot
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_chatbot.py
