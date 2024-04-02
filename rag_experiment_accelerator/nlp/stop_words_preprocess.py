import spacy

from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.nlp.preprocess import Preprocess

logger = get_logger(__name__)


class StopWordsPreprocess(Preprocess):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")

    def sentence_tokenize(self, text):
        """
        Tokenizes a given text into sentences using spacy.

        Args:
            text (str): The text to be tokenized.

        Returns:
            list: A list of sentences extracted from the input text.
        """
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def word_tokenize(self, text):
        """
        Tokenizes the input text into individual words.

        Args:
            text (str): The text to tokenize.

        Returns:
            list: A list of individual words in the input text.
        """
        return [w.text for w in self.nlp(text)]

    def remove_stopwords(self, sentence):
        """
        Removes stopwords from a given sentence.

        Args:
            sentence (str): The sentence to remove stopwords from.

        Returns:
            str: The sentence with stopwords removed.
        """
        doc = self.nlp(sentence)
        filtered_tokens = [token for token in doc if not token.is_stop]

        return " ".join([token.text for token in filtered_tokens])

    def lemmatize(self, text):
        """
        Lemmatizes the input text using the WordNet lemmatizer.

        Args:
            text (str): The text to lemmatize.

        Returns:
            str: The lemmatized text.
        """
        doc = self.nlp(text)
        return " ".join([token.lemma_ for token in doc])

    def preprocess(self, text):
        """
        Preprocesses the input text by converting it to lowercase, removing punctuation and tags, removing stopwords, and tokenizing the words.

        Args:
            text (str): The input text to preprocess.

        Returns:
            list: A list of preprocessed words.
        """
        lower_text = self.to_lower(text).strip()
        sentence_tokens = self.sentence_tokenize(lower_text)
        word_list = []
        for each_sent in sentence_tokens:
            clean_text = self.remove_punct(each_sent)
            clean_text = self.remove_Tags(clean_text)
            clean_text = self.remove_stopwords(clean_text)
            word_tokens = self.word_tokenize(clean_text)
            for i in word_tokens:
                word_list.append(i)
        return word_list
