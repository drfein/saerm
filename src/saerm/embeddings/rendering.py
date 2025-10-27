from __future__ import annotations

from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer

Message = Dict[str, str]


def coerce_message(message: Dict[str, Any]) -> Message:
    role = str(message.get("role", "assistant")).lower()
    content = message.get("content")
    if content is None:
        if "text" in message:
            content = message["text"]
        elif "value" in message:
            content = message["value"]
    return {"role": role or "assistant", "content": "" if content is None else str(content)}


def ensure_chat_messages(messages: Any, prompt: Optional[str]) -> List[Message]:
    if isinstance(messages, list):
        if messages and isinstance(messages[0], dict) and "role" in messages[0]:
            normalized = [coerce_message(entry) for entry in messages]
        else:
            normalized = [coerce_message({"role": "assistant", "content": item}) for item in messages]
    elif isinstance(messages, dict):
        normalized = [coerce_message(messages)]
    else:
        normalized = [coerce_message({"role": "assistant", "content": messages})]

    if prompt and not any(msg["role"] == "user" for msg in normalized):
        normalized.insert(0, {"role": "user", "content": str(prompt)})
    return normalized


def render_messages(messages: List[Message], tokenizer: AutoTokenizer) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
    parts = [f"[{msg.get('role', 'assistant')}]: {msg.get('content', '')}" for msg in messages]
    return "\n".join(parts)
