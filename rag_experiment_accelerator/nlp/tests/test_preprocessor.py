from unittest.mock import MagicMock, patch

from rag_experiment_accelerator.nlp.preprocess import Preprocess


@patch("rag_experiment_accelerator.nlp.preprocess.load")
def test_sentence_tokenize(mock_nlp):
    mock_sent_1 = MagicMock()
    mock_sent_1.text = "This is a sentence."
    mock_sent_2 = MagicMock()
    mock_sent_2.text = "This is another sentence.   "
    mock_nlp().return_value.sents = [mock_sent_1, mock_sent_2]
    preprocessor = Preprocess(True)
    expected = ["This is a sentence.", "This is another sentence."]

    actual = preprocessor.sentence_tokenize("text is mocked")

    assert actual == expected


@patch("rag_experiment_accelerator.nlp.preprocess.load")
def test_word_tokenize(mock_nlp):
    mock_sent_1 = MagicMock()
    mock_sent_1.text = "This"
    mock_sent_2 = MagicMock()
    mock_sent_2.text = "is"
    mock_sent_3 = MagicMock()
    mock_sent_3.text = "a"
    mock_sent_4 = MagicMock()
    mock_sent_4.text = "sentence"
    mock_nlp().return_value = [mock_sent_1, mock_sent_2, mock_sent_3, mock_sent_4]
    preprocessor = Preprocess(True)
    expected = ["This", "is", "a", "sentence"]
    actual = preprocessor.word_tokenize("text is mocked")
    assert actual == expected


@patch("rag_experiment_accelerator.nlp.preprocess.load")
def test_remove_stopwords(mock_nlp):
    mock_token_1 = MagicMock()
    mock_token_1.text = "This"
    mock_token_1.is_stop = True
    mock_token_2 = MagicMock()
    mock_token_2.text = "is"
    mock_token_2.is_stop = True
    mock_token_3 = MagicMock()
    mock_token_3.text = "a"
    mock_token_3.is_stop = True
    mock_token_4 = MagicMock()
    mock_token_4.text = "sentence"
    mock_token_4.is_stop = False
    mock_token_5 = MagicMock()
    mock_token_5.text = "."
    mock_token_5.is_stop = False
    mock_nlp().return_value = [
        mock_token_1,
        mock_token_2,
        mock_token_3,
        mock_token_4,
        mock_token_5,
    ]

    preprocessor = Preprocess(True)
    sentence = "This is a sentence."
    expected = "sentence ."
    actual = preprocessor.remove_stop_words(sentence)
    assert actual == expected


@patch("rag_experiment_accelerator.nlp.preprocess.load")
def test_lemmatize(mock_nlp):
    mock_token_1 = MagicMock()
    mock_token_1.lemma_ = "kite"
    mock_token_1.is_stop = True
    mock_token_2 = MagicMock()
    mock_token_2.lemma_ = "baby"
    mock_token_2.is_stop = True
    mock_token_3 = MagicMock()
    mock_token_3.lemma_ = "dog"
    mock_token_3.is_stop = True
    mock_token_4 = MagicMock()
    mock_token_4.lemma_ = "fly"
    mock_token_4.is_stop = False
    mock_nlp().return_value = [mock_token_1, mock_token_2, mock_token_3, mock_token_4]
    preprocessor = Preprocess(True)
    text = "kites babies dogs flying"
    expected = "kite baby dog fly"
    assert preprocessor.lemmatize(text) == expected


@patch("rag_experiment_accelerator.nlp.preprocess.load")
def test_remove_punct(mock_nlp):
    preprocessor = Preprocess(True)
    text = """this!" is*+,-. /a#$ sentence%& with'() a:;<= lot>?@[ of\\]^_ punctuation`{|}~"""
    expected = "this is a sentence with a lot of punctuation"
    assert preprocessor.remove_punctuation(text) == expected
