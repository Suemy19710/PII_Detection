## This app.py run for streamlit -> run: streamlit run app.py
import streamlit as st
import torch
from parse_txt import parse_txt
from pii_detector import detect_pii
from reporter import quick_summary

st.set_page_config(page_title="PII Detector", layout="centered")

st.title("🔍 PII Detection Tool")

mode = st.radio(
    "Choose input type:",
    ["Direct Text Input", "Upload .txt File"]
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