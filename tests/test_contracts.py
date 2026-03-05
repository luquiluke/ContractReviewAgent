from app.contracts import CONTRACT_QUESTIONS

_EXPECTED_TYPES = [
    "Employment Contract",
    "NDA / Confidentiality",
    "Freelance / Service Agreement",
    "Vendor / SaaS Agreement",
    "Assignment Letter",
    "Partnership Agreement",
    "Lease / Rental Agreement",
    "Loan / Financing Agreement",
]


def test_eight_contract_types():
    assert len(CONTRACT_QUESTIONS) == 8
    for name in _EXPECTED_TYPES:
        assert name in CONTRACT_QUESTIONS, f"Missing contract type: {name}"


def test_section_structure():
    for contract_type, sections in CONTRACT_QUESTIONS.items():
        for i, sec in enumerate(sections):
            assert set(sec.keys()) == {"section", "question"}, (
                f"{contract_type}[{i}] has unexpected keys: {set(sec.keys())}"
            )
            assert sec["section"], f"{contract_type}[{i}] 'section' is empty"
            assert sec["question"], f"{contract_type}[{i}] 'question' is empty"


def test_assignment_letter_has_seven_sections():
    assert len(CONTRACT_QUESTIONS["Assignment Letter"]) == 7


def test_other_types_have_four_to_five_sections():
    for contract_type, sections in CONTRACT_QUESTIONS.items():
        if contract_type == "Assignment Letter":
            continue
        count = len(sections)
        assert 4 <= count <= 5, (
            f"{contract_type} has {count} sections (expected 4-5)"
        )
