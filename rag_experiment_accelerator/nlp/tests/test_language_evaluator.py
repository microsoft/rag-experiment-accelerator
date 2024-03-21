from unittest.mock import patch, Mock

from rag_experiment_accelerator.nlp.language_evaluator import LanguageEvaluator

documents = [
    "This is a test.",
    "C'est un test.",
    "Dies ist ein Test.",
    "Questa è una prova.",
]
detect_language_response = {
    "name": "English",
    "iso6391_name": "en",
    "confidence_score": 1,
}
detect_languages_response = [
    {
        "name": "English",
        "iso6391_name": "en",
        "confidence_score": 1,
    },
    {
        "name": "French",
        "iso6391_name": "fr",
        "confidence_score": 1,
    },
    {
        "name": "German",
        "iso6391_name": "de",
        "confidence_score": 1,
    },
    {
        "name": "Italian",
        "iso6391_name": "it",
        "confidence_score": 1,
    },
]


def test_language_evaluator_init():
    language_evaluator = LanguageEvaluator(Mock(), "en-us", "en", "", 0.77)
    assert language_evaluator.query_language == "en-us"
    assert language_evaluator.country_hint == "us"
    assert language_evaluator.max_content_length == 50000
    assert language_evaluator.confidence_threshold == 0.77
    assert language_evaluator.default_language == "en"


def test_detect_language():
    with patch(
        "rag_experiment_accelerator.nlp.language_evaluator.LanguageEvaluator"
    ) as language_evaluator:
        language_evaluator.detect_language.return_value = detect_language_response
        primary_language = language_evaluator.detect_language("This is a test.")
        assert primary_language.get("name") == "English"
        assert primary_language.get("iso6391_name") == "en"
        assert primary_language.get("confidence_score") == 1


def test_detect_languages():
    with patch(
        "rag_experiment_accelerator.nlp.language_evaluator.LanguageEvaluator"
    ) as language_evaluator:
        response = language_evaluator.detect_language(
            documents
        ).return_value = detect_languages_response

        for i, doc in enumerate(detect_languages_response):
            assert doc["name"] == response[i].get("name")
            assert doc["iso6391_name"] == response[i].get("iso6391_name")
            assert doc["confidence_score"] == response[i].get("confidence_score")


def test_is_confident_returns_certainty():
    with patch.object(
        LanguageEvaluator, "detect_language", create=True
    ) as language_evaluator:
        language_evaluator.detect_language.return_value = detect_language_response
        language_evaluator.is_confident("This is a test.")
        language_evaluator.is_confident.assert_called()
        language_evaluator.is_confident.assert_called_with("This is a test.")


def test_is_language_match():
    with patch.object(
        LanguageEvaluator, "detect_language", create=True
    ) as language_evaluator:
        language_evaluator.is_language_match("C'est un test.", "fr")
        language_evaluator.is_language_match.assert_called()
        language_evaluator.is_language_match.assert_called_with("C'est un test.", "fr")


def test_check_string():
    language_evaluator = LanguageEvaluator(Mock())
    assert language_evaluator.check_string("This is string")
