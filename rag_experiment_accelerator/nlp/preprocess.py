import re
from string import punctuation
from typing import Union
from spacy import load

from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


class Preprocess:
    __enabled: bool

    def __init__(self, enabled=False):
        self.__enabled = enabled
        if self.__enabled:
            try:
                self.nlp = load("en_core_web_lg")
            except OSError:
                logger.info("Downloading spacy language model: en_core_web_lg")
                from spacy.cli import download

                download("en_core_web_lg")
                self.nlp = load("en_core_web_lg")

    def preprocess(self, text) -> Union[str, list[str]]:
        """
        Preprocess the input text by converting it to lowercase, removing punctuation and tags, removing stop words, and tokenizing the words.

        Args:
            text (str): The input text to preprocess (if enabled).

        Returns:
            Union[str, list[str]]:  If enabled - list of preprocessed words otherwise the original text.
        """
        if self.__enabled:
            lower_text = self.to_lower(text).strip()
            sentence_tokens = self.sentence_tokenize(lower_text)
            word_list = []
            for each_sent in sentence_tokens:
                clean_text = self.remove_punctuation(each_sent)
                clean_text = self.remove_tags(clean_text)
                clean_text = self.remove_stop_words(clean_text)
                word_tokens = self.word_tokenize(clean_text)
                for i in word_tokens:
                    word_list.append(i)
            return word_list
        else:
            return text

    def to_lower(self, text):
        """
        Convert all characters in the given text to lowercase.

        Args:
            text (str): The text to convert to lowercase.

        Returns:
            str: The converted text in lowercase.
        """
        return text.lower()

    def remove_spaces(self, text):
        """
        Removes leading and trailing spaces from a string.

        Args:
            text (str): The string to remove spaces from.

        Returns:
            str: The input string with leading and trailing spaces removed.
        """
        return text.strip()

    def remove_punctuation(self, text):
        """
        Removes all punctuation from the given text and returns the result.

        Args:
            text (str): The text to remove punctuation from.

        Returns:
            str: The text with all punctuation removed.
        """
        return "".join(c for c in text if c not in punctuation)

    def remove_tags(self, text):
        """
        Removes HTML tags from the given text.

        Args:
            text (str): The text to remove HTML tags from.

        Returns:
            str: The cleaned text with HTML tags removed.
        """
        cleaned_text = re.sub("<[^<]+?>", "", text)
        return cleaned_text

    def sentence_tokenize(self, text):
        """
        Tokenize a given text into sentences using spacy.

        Args:
            text (str): The text to be tokenized.

        Returns:
            list: A list of sentences extracted from the input text.
        """
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]

    def word_tokenize(self, text):
        """
        Tokenize the input text into individual words.

        Args:
            text (str): The text to tokenize.

        Returns:
            list: A list of individual words in the input text.
        """
        return [w.text for w in self.nlp(text)]

    def remove_stop_words(self, sentence):
        """
        Removes stop words from a given sentence.

        Args:
            sentence (str): The sentence to remove stop words from.

        Returns:
            str: The sentence with stop words removed.
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
