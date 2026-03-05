from unittest.mock import patch, MagicMock

import pytest

from app.review import analyze_contract


def _mock_response(text="The contract provides standard terms."):
    r = MagicMock()
    r.content = text
    return r


def test_analyze_returns_results():
    with patch("app.review._LANGCHAIN_AVAILABLE", True), \
         patch("app.review._llm") as mock_llm:
        mock_llm.invoke.return_value = _mock_response()
        results = analyze_contract("sample contract text", "Employment Contract")
    assert isinstance(results, list)
    assert len(results) == 5
    for item in results:
        assert "section" in item
        assert "summary" in item


def test_not_addressed_passthrough():
    not_addressed = "This contract does not address Termination Conditions."
    with patch("app.review._LANGCHAIN_AVAILABLE", True), \
         patch("app.review._llm") as mock_llm:
        mock_llm.invoke.return_value = _mock_response(not_addressed)
        results = analyze_contract("sample contract text", "Employment Contract")
    assert results[0]["summary"] == not_addressed


def test_progress_callback_called():
    calls = []

    def cb(idx, total, name):
        calls.append((idx, total, name))

    with patch("app.review._LANGCHAIN_AVAILABLE", True), \
         patch("app.review._llm") as mock_llm:
        mock_llm.invoke.return_value = _mock_response()
        analyze_contract("text", "Employment Contract", progress_callback=cb)

    assert len(calls) == 5


def test_raises_when_unavailable():
    with patch("app.review._LANGCHAIN_AVAILABLE", False):
        with pytest.raises(RuntimeError):
            analyze_contract("text", "Employment Contract")
