def extract_text_from_openai_json(data: dict) -> str:
    texts = []

    # Case 1: ChatGPT export (mapping)
    if "mapping" in data and isinstance(data["mapping"], dict):
        for node in data["mapping"].values():
            msg = node.get("message")
            if not msg:
                continue

            content = msg.get("content", {})
            parts = content.get("parts", [])

            for part in parts:
                if isinstance(part, str):
                    texts.append(part)

    # Case 2: API-style messages
    elif "messages" in data and isinstance(data["messages"], list):
        for msg in data["messages"]:
            content = msg.get("content")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and "text" in c:
                        texts.append(c["text"])

    return "\n".join(texts)