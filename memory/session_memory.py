import uuid
from datetime import datetime


class SessionMemory:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self._messages: list[dict] = []

    def add_message(self, role: str, content: str):
        self._messages.append({"role": role, "content": content})

    def get_messages_for_api(self) -> list[dict]:
        """Returns messages in Anthropic API format (alternating user/assistant)."""
        return [{"role": m["role"], "content": m["content"]} for m in self._messages]

    def get_messages(self) -> list[dict]:
        return list(self._messages)

    def clear(self):
        self._messages = []
        self.session_id = str(uuid.uuid4())
        self.created_at = datetime.now()

    def summary_text(self) -> str:
        lines = [f"[{m['role'].upper()}] {m['content']}" for m in self._messages]
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._messages)
