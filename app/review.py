"""
app/review.py — PDF text extraction for Scrutiny.

Phase 1: extract_text_from_pdf only.
Phase 2 will add: strip_pii (uses Presidio).
Phase 3 will add: analyze_contract (uses LangChain + OpenAI).
"""
import io

from pypdf import PdfReader


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
