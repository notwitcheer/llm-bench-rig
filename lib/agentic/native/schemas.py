"""Convert the rig's mock_tools TOOLS list into OpenAI function-calling schemas.

Each TOOLS entry is {name, signature 'fn(a: type, b: type) -> ret', description}.
We parse the signature's params into JSON-schema string/number properties (all required).
"""

_TYPE = {"str": "string", "float": "number", "int": "integer", "bool": "boolean"}


def _parse_params(signature: str) -> dict:
    inner = signature[signature.index("(") + 1: signature.index(")")]
    props, required = {}, []
    for part in [p.strip() for p in inner.split(",") if p.strip()]:
        name, _, typ = part.partition(":")
        name = name.strip()
        props[name] = {"type": _TYPE.get(typ.strip(), "string")}
        required.append(name)
    return {"type": "object", "properties": props, "required": required}


def to_openai_tools(tools: list) -> list:
    return [
        {"type": "function", "function": {
            "name": t["name"], "description": t["description"],
            "parameters": _parse_params(t["signature"]),
        }}
        for t in tools
    ]
