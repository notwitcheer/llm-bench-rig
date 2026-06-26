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
        if r.status_code != 200:
            # llama-server returns 500 when it can't parse a model's tool-call JSON
            # (e.g. unescaped newlines in a multi-line bash arg) or when the request
            # exceeds the context window. Surface it as a recoverable message instead of
            # raising and crashing the whole trajectory — the loop feeds the error back so
            # the model can retry (parse error) or ends cleanly (context overflow).
            try:
                err = r.json().get("error", {}).get("message", r.text)
            except Exception:
                err = r.text
            return {"role": "assistant", "content": "", "_error": str(err)[:600], "_tokens": 0}
        data = r.json()
        msg = data["choices"][0]["message"]
        msg["_tokens"] = data.get("usage", {}).get("completion_tokens", 0)
        return msg
