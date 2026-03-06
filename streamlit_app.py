import os
import re
from pathlib import Path

import streamlit as st

from app.review import extract_text_from_pdf, extract_and_strip, _build_prompt, _get_llm
from app.contracts import CONTRACT_QUESTIONS
from app.exporters import build_excel, build_pdf

def _safe(text: str) -> str:
    """Escape dollar signs so Streamlit doesn't render them as LaTeX math."""
    return text.replace("$", r"\$")


_PII_LABELS = {
    "PERSON": "name",
    "EMAIL_ADDRESS": "email address",
    "PHONE_NUMBER": "phone number",
    "IBAN_CODE": "IBAN",
    "DATE_TIME": "date",
}

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
        contract_text, pii_manifest = extract_and_strip(uploaded_file.getvalue())
        pii_count = sum(pii_manifest.values())
        st.success(f"Contract loaded — {len(contract_text):,} characters extracted.")
        if pii_count > 0:
            lines = [f"{pii_count} item{'s' if pii_count != 1 else ''} removed before sending to AI:"]
            for entity, count in pii_manifest.items():
                label = _PII_LABELS.get(entity, entity.lower().replace("_", " "))
                lines.append(f"- {count} {label}{'s' if count != 1 else ''}")
            st.info("\n".join(lines))
        st.session_state["contract_text"] = contract_text
        st.session_state["pii_manifest"] = pii_manifest
    except ValueError as e:
        st.error(str(e))

# ── Main: Analysis ────────────────────────────────────────────────────
if st.session_state.get("contract_text"):

    def _clear_analysis():
        st.session_state.pop("analysis_results", None)
        st.session_state.pop("analysis_contract_type", None)

    contract_type = st.selectbox(
        "Contract type",
        options=list(CONTRACT_QUESTIONS.keys()),
        key="contract_type",
        on_change=_clear_analysis,
    )

    has_api_key = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    is_analyzing = st.session_state.get("_analyzing", False)

    if st.button("Review contract", disabled=not has_api_key or is_analyzing):
        st.session_state["_analyzing"] = True
        try:
            sections = CONTRACT_QUESTIONS[contract_type]
            placeholders = [st.empty() for _ in sections]
            progress_bar = st.progress(0)

            results = []

            for i, section_info in enumerate(sections):
                section_name = section_info["section"]
                progress_bar.progress(
                    i / len(sections),
                    text=f"Analyzing section {i + 1} of {len(sections)}: {section_name}...",
                )
                prompt_text = st.session_state["contract_text"]
                raw_summary = _get_llm().invoke(
                    _build_prompt(prompt_text, section_name, section_info["question"])
                ).content
                raw_summary = re.sub(r"`([^`]*)`", r"\1", raw_summary)
                results.append({"section": section_name, "summary": raw_summary})

                with placeholders[i].container():
                    st.markdown(f"**{section_name}**")
                    if raw_summary.lower().startswith("this contract does not address"):
                        st.caption(_safe(raw_summary))
                    else:
                        st.write(_safe(raw_summary))
                progress_bar.progress(
                    (i + 1) / len(sections),
                    text=f"Analyzing section {i + 1} of {len(sections)}: {section_name}...",
                )

            progress_bar.empty()
            st.session_state["analysis_results"] = results
            st.session_state["analysis_contract_type"] = contract_type
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download Excel",
                    data=build_excel(results),
                    file_name="scrutiny-review.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            with col2:
                st.download_button(
                    label="Download PDF",
                    data=build_pdf(results, contract_type),
                    file_name="scrutiny-review.pdf",
                    mime="application/pdf",
                )
        except Exception as e:
            st.error(f"Analysis failed: {e}")
        finally:
            st.session_state["_analyzing"] = False

    elif (
        st.session_state.get("analysis_results")
        and st.session_state.get("analysis_contract_type") == contract_type
    ):
        results = st.session_state["analysis_results"]
        for row in results:
            st.markdown(f"**{row['section']}**")
            if row["summary"].lower().startswith("this contract does not address"):
                st.caption(_safe(row["summary"]))
            else:
                st.write(_safe(row["summary"]))
            st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Download Excel",
                data=build_excel(results),
                file_name="scrutiny-review.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with col2:
            st.download_button(
                label="Download PDF",
                data=build_pdf(results, contract_type),
                file_name="scrutiny-review.pdf",
                mime="application/pdf",
            )

# ── Main: Legal disclaimer (always visible — not inside any conditional) ──
st.divider()
st.caption(
    "Scrutiny is not a law firm and does not provide legal advice. "
    "AI-generated summaries may be incomplete or inaccurate. "
    "Consult a qualified attorney before acting on any information."
)
