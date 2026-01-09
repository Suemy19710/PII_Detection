import json

def parse_openai_json(file_content):
    """Parse OpenAI chat history JSON format"""
    try:
        data = json.loads(file_content)
        conversations = []

        # Handle different JSON structures
        if isinstance(data, list):
            conversations_data = data
        elif isinstance(data, dict):
            # Check if it's the newer OpenAI format
            if "conversations" in data:
                conversations_data = data["conversations"]
            else:
                conversations_data = [data]
        else:
            raise ValueError("Invalid JSON structure")

        for conv in conversations_data:
            title = conv.get("title", "Untitled")
            create_time = conv.get("create_time", conv.get("timestamp", "Unknown"))

            messages = []

            # Handle different conversation structures
            mapping = conv.get("mapping", {})
            if mapping:
                # OpenAI format with mapping
                for node in mapping.values():
                    msg = node.get("message")
                    if not msg:
                        continue

                    if msg.get("author", {}).get("role") != "user":
                        continue

                    parts = msg.get("content", {}).get("parts", [])
                    text = extract_text(parts)

                    if text and text.strip():
                        messages.append({
                            "text": text.strip(),
                            "timestamp": create_time
                        })
            else:
                # Simple message list format
                messages_list = conv.get("messages", [])
                for msg in messages_list:
                    if msg.get("role") == "user":
                        text = extract_text([msg.get("content", "")])
                        if text and text.strip():
                            messages.append({
                                "text": text.strip(),
                                "timestamp": create_time
                            })

            if messages:
                conversations.append({
                    "title": title,
                    "messages": messages
                })

        return conversations
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")
    except Exception as e:
        raise ValueError(f"Error parsing OpenAI JSON: {e}")


def extract_text(parts):
    """Extract text from message parts safely"""
    texts = []
    if not parts:
        return ""
    
    for p in parts:
        if isinstance(p, str):
            texts.append(p)
        elif isinstance(p, dict):
            # Handle structured parts safely
            if "text" in p and isinstance(p["text"], str):
                texts.append(p["text"])
            elif "content" in p and isinstance(p["content"], str):
                texts.append(p["content"])
            elif "text" in p and isinstance(p["text"], dict):
                # Handle nested text dict
                nested_text = extract_text([p["text"]])
                if nested_text:
                    texts.append(nested_text)
        elif isinstance(p, list):
            # Recursively handle lists
            nested_text = extract_text(p)
            if nested_text:
                texts.append(nested_text)
    
    return " ".join(texts).strip()