from rag_experiment_accelerator.nlp.preprocess import Preprocess

# Bug with vscode and pytest
import sys
sys.path.append('/workspaces/rag-experiment-accelerator/nlp')


def test_sentence_tokenize():
    preprocessor = Preprocess()
    text = "This is a sentence. This is another sentence."
    expected = ["This is a sentence.", "This is another sentence."]
    assert preprocessor.sentence_tokenize(text) == expected

def test_word_tokenize():
    preprocessor = Preprocess()
    text = "This is a sentence."
    expected = ["This", "is", "a", "sentence", "."]
    assert preprocessor.word_tokenize(text) == expected

def test_remove_stopwords():
    preprocessor = Preprocess()
    sentence = "This is a sentence."
    expected = "sentence ."
    assert preprocessor.remove_stopwords(sentence) == expected

def test_lemmatize():
    preprocessor = Preprocess()
    text = "kites babies dogs flying smiling driving died tried feet"
    expected = "kite baby dog fly smile drive die try foot"
    assert preprocessor.lemmatize(text) == expected
