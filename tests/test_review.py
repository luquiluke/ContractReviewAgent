import io
import pytest
from app.review import extract_text_from_pdf


# Minimal valid PDF with one blank page — no text content stream.
# Source: RESEARCH.md scanned PDF unit test pattern.
BLANK_PDF_BYTES = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
trailer<</Size 4/Root 1 0 R>>
startxref
190
%%EOF"""


def test_scanned_pdf_raises_value_error():
    """Blank PDF with no text layer raises ValueError with 'scanned' in message."""
    with pytest.raises(ValueError, match="scanned"):
        extract_text_from_pdf(BLANK_PDF_BYTES)


def test_scanned_pdf_error_message_is_user_friendly():
    """Error message should be intelligible to a non-technical user."""
    with pytest.raises(ValueError) as exc_info:
        extract_text_from_pdf(BLANK_PDF_BYTES)
    msg = str(exc_info.value)
    assert "scanned" in msg.lower() or "image" in msg.lower()
    assert len(msg) > 20  # not a bare "ValueError"


def test_valid_pdf_returns_string():
    """A real text-layer PDF returns a non-empty string longer than 100 chars."""
    # Build a minimal but real text-layer PDF using pypdf's writer
    from pypdf import PdfWriter
    from reportlab.pdfgen import canvas as rl_canvas

    # If reportlab is not available, skip with a note
    pytest.importorskip("reportlab", reason="reportlab not installed — skipping valid PDF test")

    import io as _io
    buf = _io.BytesIO()
    c = rl_canvas.Canvas(buf)
    c.drawString(72, 720, "This is a test contract. " * 10)
    c.save()
    buf.seek(0)

    result = extract_text_from_pdf(buf.read())
    assert isinstance(result, str)
    assert len(result.strip()) >= 100
