from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class UserState:
    tasks: List[Dict] = field(default_factory=list)
    next_id: int = 1

    def add_task(self, text: str, due: Optional[str] = None) -> int:
        task = {"id": self.next_id, "text": text.strip(), "due": due}
        self.tasks.append(task)
        self.next_id += 1
        return task["id"]

    def list_tasks(self) -> List[Dict]:
        return list(self.tasks)

    def delete_task(self, task_id: int) -> bool:
        before = len(self.tasks)
        self.tasks = [t for t in self.tasks if t["id"] != task_id]
        return len(self.tasks) != before
