from app.pii import strip_pii, REDACTED_ENTITY_TYPES


def test_entity_list_contains_exactly_five_types():
    assert set(REDACTED_ENTITY_TYPES) == {"PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "IBAN_CODE", "DATE_TIME"}
    assert len(REDACTED_ENTITY_TYPES) == 5


def test_empty_string_returns_empty():
    redacted, manifest = strip_pii("")
    assert redacted == ""
    assert manifest == {}


def test_email_redacted():
    redacted, manifest = strip_pii("Send to john@acme.com for details")
    assert "john@acme.com" not in redacted
    assert "[EMAIL]" in redacted
    assert manifest.get("EMAIL_ADDRESS", 0) == 1


def test_phone_redacted():
    redacted, manifest = strip_pii("Call +1-800-555-0199 for support")
    assert "+1-800-555-0199" not in redacted
    assert "[PHONE]" in redacted
    assert manifest.get("PHONE_NUMBER", 0) == 1


def test_iban_redacted():
    redacted, manifest = strip_pii("Bank account: GB29NWBK60161331926819")
    assert "GB29NWBK60161331926819" not in redacted
    assert "[IBAN]" in redacted
    assert manifest.get("IBAN_CODE", 0) == 1


def test_date_redacted():
    redacted, manifest = strip_pii("Effective date: 2024-01-15")
    assert "2024-01-15" not in redacted
    assert "[DATE]" in redacted
    assert manifest.get("DATE_TIME", 0) == 1


def test_person_redacted_mid_sentence():
    # Use name mid-sentence — en_core_web_sm PERSON detection unreliable at sentence start
    redacted, manifest = strip_pii("The contractor John Smith agrees to the terms")
    assert "John Smith" not in redacted
    assert "[PERSON]" in redacted
    assert manifest.get("PERSON", 0) >= 1


def test_all_five_entities_in_one_string():
    sample = (
        "The contractor John Smith (john@acme.com, +1-800-555-0199) "
        "signed on 2024-01-15. IBAN: GB29NWBK60161331926819."
    )
    redacted, manifest = strip_pii(sample)
    assert "John Smith" not in redacted
    assert "john@acme.com" not in redacted
    assert "+1-800-555-0199" not in redacted
    assert "2024-01-15" not in redacted
    assert "GB29NWBK60161331926819" not in redacted
    for entity_type in REDACTED_ENTITY_TYPES:
        assert manifest.get(entity_type, 0) >= 1, f"Missing: {entity_type}"


def test_manifest_counts_multiple_occurrences():
    redacted, manifest = strip_pii("Emails: alice@test.com and bob@test.com")
    assert manifest.get("EMAIL_ADDRESS", 0) >= 2


def test_returns_tuple_of_str_and_dict():
    result = strip_pii("Hello world")
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], str)
    assert isinstance(result[1], dict)
