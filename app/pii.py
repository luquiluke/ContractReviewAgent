from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

_NLP_CONFIG = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
}
_provider = NlpEngineProvider(nlp_configuration=_NLP_CONFIG)
_nlp_engine = _provider.create_engine()
_analyzer = AnalyzerEngine(nlp_engine=_nlp_engine, supported_languages=["en"])
_anonymizer = AnonymizerEngine()

REDACTED_ENTITY_TYPES = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "IBAN_CODE", "DATE_TIME"]

_OPERATORS = {
    "PERSON":        OperatorConfig("replace", {"new_value": "[PERSON]"}),
    "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL]"}),
    "PHONE_NUMBER":  OperatorConfig("replace", {"new_value": "[PHONE]"}),
    "IBAN_CODE":     OperatorConfig("replace", {"new_value": "[IBAN]"}),
    "DATE_TIME":     OperatorConfig("replace", {"new_value": "[DATE]"}),
}


def strip_pii(text: str) -> tuple[str, dict[str, int]]:
    if not text:
        return text, {}
    analyzer_results = _analyzer.analyze(
        text=text, entities=REDACTED_ENTITY_TYPES, language="en"
    )
    anonymized = _anonymizer.anonymize(
        text=text, analyzer_results=analyzer_results, operators=_OPERATORS
    )
    manifest: dict[str, int] = {}
    for result in analyzer_results:
        manifest[result.entity_type] = manifest.get(result.entity_type, 0) + 1
    return anonymized.text, manifest
