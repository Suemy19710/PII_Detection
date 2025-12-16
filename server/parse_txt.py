def parse_txt(text: str, title="Direct Text Input"):
    # Normalize raw text into conversation format
    return [{
        "id": "text_input",
        "title": title,
        "messages": [{
            "role": "user",
            "text": text
        }]
    }]
