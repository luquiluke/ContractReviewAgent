"""
app/review.py — PDF text extraction for Scrutiny.

Phase 1: extract_text_from_pdf only.
Phase 2 will add: strip_pii (uses Presidio).
Phase 3 will add: analyze_contract (uses LangChain + OpenAI).
"""
import io

from pypdf import PdfReader

from app.pii import strip_pii
from app.contracts import CONTRACT_QUESTIONS

try:
    from langchain_openai import ChatOpenAI
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False

def _get_llm():
    if not _LANGCHAIN_AVAILABLE:
        return None
    import os
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract text from a PDF given its raw bytes.

    Uses in-memory extraction — no temp files written to disk.

    Args:
        pdf_bytes: Raw bytes of the PDF file.

    Returns:
        Full concatenated text from all pages.

    Raises:
        ValueError: If extracted text is < 100 chars after stripping whitespace.
                    This indicates a scanned or image-only document.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""  # coerce None for blank pages

    if len(text.strip()) < 100:
        raise ValueError(
            "Could not extract text from this PDF. "
            "It appears to be a scanned or image-only document. "
            "Please upload a PDF with a text layer."
        )
    return text


def extract_and_strip(pdf_bytes: bytes) -> tuple[str, dict[str, int]]:
    text = extract_text_from_pdf(pdf_bytes)
    return strip_pii(text)


def _build_prompt(contract_text: str, section: str, question: str) -> str:
    return (
        f"You are a contract analyst. Review the following contract and answer ONE question.\n\n"
        f"CONTRACT:\n{contract_text}\n\n"
        f"QUESTION: {question}\n\n"
        f"Respond in 2-4 plain-English sentences. "
        f"Do not use markdown, backticks, bold, or any special formatting — plain text only. "
        f"If the contract does not address this topic, say exactly: "
        f"'This contract does not address {section}.' "
        f"Do not speculate or infer from general knowledge."
    )


def analyze_contract(
    contract_text: str,
    contract_type: str,
    progress_callback=None,
) -> list[dict]:
    if not _LANGCHAIN_AVAILABLE:
        raise RuntimeError(
            "LangChain is not installed. Install langchain-openai to use analysis."
        )
    sections = CONTRACT_QUESTIONS[contract_type]
    total = len(sections)
    results = []
    for i, section_info in enumerate(sections):
        section = section_info["section"]
        question = section_info["question"]
        prompt = _build_prompt(contract_text, section, question)
        summary = _get_llm().invoke(prompt).content
        results.append({"section": section, "summary": summary})
        if progress_callback is not None:
            progress_callback(i, total, section)
    return results
