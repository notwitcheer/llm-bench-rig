"""Thin OpenAI /chat/completions client for llama-server. Returns the assistant message
dict with an extra _tokens field (completion_tokens) for the token-efficiency axis."""
import httpx


class LlamaClient:
    def __init__(self, base="http://127.0.0.1:8090/v1", model="local", timeout=120,
                 think: bool = True):
        self.url = f"{base}/chat/completions"
        self.model = model
        self.timeout = timeout
        self.think = think
        self.last_reasoning_tokens = 0
        self.last_reasoning_content = ""

    def chat(self, messages, tools):
        payload = {"model": self.model, "messages": messages, "temperature": 0}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if not self.think:
            # Suppress reasoning across template families in one payload (mirrors
            # lib/evals/base.py's LLMClient): Gemma/Qwen read `enable_thinking`,
            # Cohere/North (cohere2moe) read `reasoning`/`reasoning_effort`. Each
            # template uses the key it knows and ignores the rest.
            payload["chat_template_kwargs"] = {
                "enable_thinking": False,
                "reasoning": False,
                "reasoning_effort": "none",
            }
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
            self.last_reasoning_tokens = 0
            self.last_reasoning_content = ""
            return {"role": "assistant", "content": "", "_error": str(err)[:600], "_tokens": 0}
        data = r.json()
        msg = data["choices"][0]["message"]
        usage = data.get("usage") or {}
        msg["_tokens"] = usage.get("completion_tokens", 0)
        # reasoning_content is present (think=True) when the server exposes model
        # "thinking" separately from the final answer; stash it for callers that want
        # it. llama-server's usage object has no reasoning_tokens field (only
        # prompt/completion/total/cached), so an explicit presence check (not
        # truthiness — a legitimately-present 0 must not trigger the estimate) falls
        # back to a chars/4 estimate over reasoning_content, mirroring the
        # usage-missing fallback in lib/evals/base.py:66-68.
        self.last_reasoning_content = msg.get("reasoning_content") or ""
        if "reasoning_tokens" in usage:
            self.last_reasoning_tokens = usage["reasoning_tokens"]
        elif self.last_reasoning_content:
            self.last_reasoning_tokens = len(self.last_reasoning_content) // 4 + 1
        else:
            self.last_reasoning_tokens = 0
        return msg
