"""Thin OpenAI /chat/completions client for llama-server. Returns the assistant message
dict with an extra _tokens field (completion_tokens) for the token-efficiency axis."""
import httpx


class LlamaClient:
    def __init__(self, base="http://127.0.0.1:8090/v1", model="local", timeout=120):
        self.url = f"{base}/chat/completions"
        self.model = model
        self.timeout = timeout

    def chat(self, messages, tools):
        payload = {"model": self.model, "messages": messages, "temperature": 0}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        r = httpx.post(self.url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        msg = data["choices"][0]["message"]
        msg["_tokens"] = data.get("usage", {}).get("completion_tokens", 0)
        return msg
