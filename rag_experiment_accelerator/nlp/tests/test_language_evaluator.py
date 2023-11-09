from unittest.mock import patch

from rag_experiment_accelerator.nlp.language_evaluator import LanguageEvaluator

documents = ["This is a test.", "C'est un test.", "Dies ist ein Test.","Questa è una prova."]
detect_language_response = [
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


def test_detect_languages_detects_all_language_content():
    with patch(
        "rag_experiment_accelerator.nlp.language_evaluator.LanguageEvaluator"
    ) as language_evaluator:
        language_evaluator.detect_language = [True, False, True]

        _, response = language_evaluator.detect_language(documents)
    
    for i, doc in enumerate(detect_language_response):
        assert doc["content"] == response[i]

    
def test_detect_language():
    language_evaluator = LanguageEvaluator()
    primary_language = language_evaluator.detect_language("This is a test.") 
    assert primary_language.name == "English"   
    assert primary_language.iso6391_name == "en"
    assert primary_language.confidence_score == 1
    primary_language = language_evaluator.detect_language("C'est un test.") 
    assert primary_language.name == "French"   
    assert primary_language.iso6391_name == "fr"
    assert primary_language.confidence_score == 1
    primary_language = language_evaluator.detect_language("Dies ist ein Test.") 
    assert primary_language.name == "German"   
    assert primary_language.iso6391_name == "de"
    assert primary_language.confidence_score == 1    


def test_detect_languages():        
    language_evaluator = LanguageEvaluator()
    response = language_evaluator.detect_language(documents)
    
    for i, doc in enumerate(detect_language_response):
        assert doc["name"] == response[i].primary_language.name
        assert doc["iso6391_name"] == response[i].primary_language.iso6391_name
        assert doc["confidence_score"] == response[i].primary_language.confidence_score


def test_is_confident_returns_certainty():
    language_evaluator = LanguageEvaluator()
    assert language_evaluator.is_confident("This is a test.") == 1
    assert language_evaluator.is_confident("C'est un test.") == 1
    assert language_evaluator.is_confident("Dies ist ein Test.") == 1


def test_is_language_match():
    language_evaluator = LanguageEvaluator()
    assert language_evaluator.is_language_match("C'est un test.", "fr") is True
    assert language_evaluator.is_language_match("This is a test.", "fr") is False
