import os
from pathlib import Path

import streamlit as st

from app.review import extract_text_from_pdf

# ── Page config (MUST be the first Streamlit call) ────────────────────
st.set_page_config(
    page_title="Scrutiny",
    page_icon="🔍",
    layout="centered",
)

# ── Theme injection ───────────────────────────────────────────────────
css_path = Path(__file__).parent / "scrutiny_theme.css"
with open(css_path) as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Sidebar: API key ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Scrutiny")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key. Not stored between sessions.",
    )
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key.strip()

# ── Main: Branding ────────────────────────────────────────────────────
st.title("Scrutiny")
st.subheader("AI-powered contract review")

# ── Main: Upload ──────────────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload a contract PDF", type=["pdf"])

if uploaded_file is not None:
    try:
        contract_text = extract_text_from_pdf(uploaded_file.getvalue())
        st.success(f"Contract loaded — {len(contract_text):,} characters extracted.")
        st.session_state["contract_text"] = contract_text
    except ValueError as e:
        st.error(str(e))

# ── Main: Legal disclaimer (always visible — not inside any conditional) ──
st.divider()
st.caption(
    "Scrutiny is not a law firm and does not provide legal advice. "
    "AI-generated summaries may be incomplete or inaccurate. "
    "Consult a qualified attorney before acting on any information."
)
