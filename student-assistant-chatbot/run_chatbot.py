from chatbot.state import UserState
from chatbot.responses import HELP_TEXT, WELCOME_TEXT
from chatbot.nlp import detect_intent, parse_add_task, parse_delete_task, FaqMatcher

def main():
    state = UserState()
    faq = FaqMatcher("data/faq.json")

    print(WELCOME_TEXT)

    while True:
        user = input("\nYou: ").strip()
        if not user:
            print("Bot: Type something, or type 'help' to see what I can do.")
            continue

        intent = detect_intent(user)

        if intent == "exit":
            print("Bot: Bye! Good luck with your studies 👋")
            break

        if intent == "help":
            print(HELP_TEXT)
            continue

        if intent == "show_tasks":
            tasks = state.list_tasks()
            if not tasks:
                print("Bot: You have no tasks yet. Try: add task: Finish lab 6 by 25 Jan")
            else:
                print("Bot: Here are your tasks:")
                for t in tasks:
                    due = f" (due: {t['due']})" if t["due"] else ""
                    print(f"  {t['id']}. {t['text']}{due}")
            continue

        if intent == "add_task":
            parsed = parse_add_task(user)
            if not parsed:
                print("Bot: I couldn't parse that. Try: add task: Submit essay by 2026-01-25")
                continue
            task_id = state.add_task(parsed["text"], parsed.get("due"))
            due_msg = f" (due: {parsed['due']})" if parsed.get("due") else ""
            print(f"Bot: Added task #{task_id} ✅{due_msg}")
            continue

        if intent == "delete_task":
            task_id = parse_delete_task(user)
            if task_id is None:
                print("Bot: Tell me which one to delete. Example: delete task 2")
                continue
            ok = state.delete_task(task_id)
            print("Bot: Deleted ✅" if ok else "Bot: I couldn't find that task number.")
            continue

        if intent == "faq":
            answer = faq.answer(user)
            if answer:
                print(f"Bot: {answer}")
            else:
                print("Bot: I’m not sure. Try asking another way, or type 'help'.")
            continue

        # Fallback
        print("Bot: I didn’t catch that. Type 'help' for options.")

if __name__ == "__main__":
    main()
