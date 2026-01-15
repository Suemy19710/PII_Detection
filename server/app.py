## This app.py run for streamlit -> run: streamlit run app.py

#TensorFlow was removed to avoid incompatibility with Python 3.12 and unnecessary dependencies. 
# The system relies purely on PyTorch for inference.
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # hide INFO + WARNING

import streamlit as st
import torch
import json
from parse_txt import parse_txt
from parse_openai_json import extract_text_from_openai_json
from pii_detector import detect_pii
from reporter import quick_summary
st.set_page_config(page_title="PII Detector", layout="centered")

st.title("🔍 PII Detection Tool")

mode = st.radio(
    "Choose input type:",
    ["Direct Text Input", "Upload .txt File", "Upload OpenAI JSON File"]
)

text_input = ""

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

elif mode == "Upload OpenAI JSON File":
    uploaded = st.file_uploader("Upload OpenAI JSON file", type=["json"])
    if uploaded is not None:
        try:
            data = json.load(uploaded)

            st.subheader("Raw JSON Preview")
            st.json(data, expanded=False)

            extracted_text = extract_text_from_openai_json(data)

            if not extracted_text.strip():
                st.warning("No readable text found, scanning full JSON as fallback")
                extracted_text = json.dumps(data, ensure_ascii=False)

            text_input = extracted_text

        except Exception as e:
            st.error(f"Error parsing JSON: {e}")

if st.button("Analyze") and text_input.strip():
    conversations = parse_txt(text_input)

    all_entities = []
    for conv in conversations:
        for msg in conv["messages"]:
            entities = detect_pii(msg["text"])
            all_entities.extend(entities)

    if not all_entities:
        st.success("No PII detected 🎉")
    else:
        summary = quick_summary(all_entities)

        st.warning("PII detected:")
        for k, v in summary.items():
            st.write(f"**{k}**")
            for val in v:
                st.code(val)
