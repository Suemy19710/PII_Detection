## This app.py run for streamlit -> run: streamlit run app.py
import streamlit as st
import pandas as pd
import json
from parse_txt import parse_txt
from pii_detector import detect_pii
from reporter import quick_summary, generate_report_json
from parse_openai_json import extract_text, parse_openai_json

st.set_page_config(page_title="PII Detector", layout="centered")
st.title("🔍 PII Detection Tool")

mode = st.radio(
    "Choose input type:",
    ["Direct Text Input", "Upload .txt File", "Upload OpenAI Chat History (.json)"]
)

text_input = None
conversations = None

# -------- INPUT --------
if mode == "Direct Text Input":
    text_input = st.text_area(
        "Enter text to scan for PII:",
        height=150,
        placeholder="My email is test@gmail.com and my phone is 0987654321"
    )

elif mode == "Upload .txt File":
    uploaded = st.file_uploader("Upload .txt file", type=["txt"])
    if uploaded:
        text_input = uploaded.read().decode("utf-8", errors="ignore")

elif mode == "Upload OpenAI Chat History (.json)":
    uploaded = st.file_uploader("Upload OpenAI chat history (.json)", type=["json"])
    if uploaded:
        try:
            conversations = parse_openai_json(
                uploaded.read().decode("utf-8")
            )
        except Exception as e:
            st.error(f"Error parsing JSON: {e}")
            conversations = None

# -------- ANALYZE --------
if st.button("Analyze"):

    all_entities = []

    # Text / TXT mode
    if text_input:
        try:
            conversations = parse_txt(text_input)
        except Exception as e:
            st.error(f"Error parsing text: {e}")
            st.stop()

    if not conversations:
        st.warning("No conversations to analyze.")
        st.stop()

    # Run PII detection
    for conv in conversations:
        for msg in conv["messages"]:
            try:
                entities = detect_pii(msg["text"])
                for e in entities:
                    e["conversation"] = conv.get("title", "Untitled")
                    e["timestamp"] = msg.get("timestamp")
                all_entities.extend(entities)
            except Exception as e:
                st.warning(f"Error detecting PII in message: {e}")
                continue

    # -------- RESULTS --------
    if not all_entities:
        st.success("No PII detected 🎉")
        st.stop()

    # Metrics - FIXED: Ensure simple scalar values
    try:
        metrics_data = {
            "Total Conversations": len(conversations),
            "Total PII Instances": len(all_entities),
            "PII Types": len(set(e["type"] for e in all_entities if "type" in e))
        }
        
        # Display metrics as individual metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Conversations", len(conversations))
        with col2:
            st.metric("Total PII Instances", len(all_entities))
        with col3:
            st.metric("PII Types", len(set(e["type"] for e in all_entities)))
        
        # Optional: Display as table using a different approach
        # metrics_df = pd.DataFrame(
        #     list(metrics_data.items()),
        #     columns=["Metric", "Value"]
        # )
        # st.table(metrics_df)
        
    except Exception as e:
        st.error(f"Error creating metrics: {e}")

    # Detailed report
    try:
        report = generate_report_json(all_entities)

        for pii_type, items in report.items():
            st.subheader(f"{pii_type} ({len(items)})")

            for item in items:
                st.markdown(
                    f"- `{item.get('value', 'N/A')}`  \n"
                    f"  🗂 **Chat:** {item.get('conversation', 'Unknown')}  \n"
                    f"  🕒 **Time:** {item.get('timestamp', 'Unknown')}"
                )
    except Exception as e:
        st.error(f"Error generating report: {e}")


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