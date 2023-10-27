import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from string import punctuation


snowball_stemmer = SnowballStemmer('english')
wordnet_lemmatizer = WordNetLemmatizer()

class Preprocess:
    def __int__(self):
        pass

    def to_lower(self, text):
            """
            Convert all characters in the given text to lowercase.

            Args:
                text (str): The text to convert to lowercase.

            Returns:
                str: The converted text in lowercase.
            """
            return text.lower()

    def remove_punct(self, text):
        """
        Removes all punctuation from the given text and returns the result.

        Args:
            text (str): The text to remove punctuation from.

        Returns:
            str: The text with all punctuation removed.
        """
        return ' '.join(c for c in text if c not in punctuation)

    def remove_Tags(self, text):
            """
            Removes HTML tags from the given text.

            Args:
                text (str): The text to remove HTML tags from.

            Returns:
                str: The cleaned text with HTML tags removed.
            """
            cleaned_text = re.sub('<[^<]+?>', '', text)
            return cleaned_text

    def sentence_tokenize(self, text):
            """
            Tokenizes a given text into sentences using the nltk.sent_tokenize method.

            Args:
                text (str): The text to be tokenized.

            Returns:
                list: A list of sentences extracted from the input text.
            """
            sent_list = []
            for w in nltk.sent_tokenize(text):
                sent_list.append(w)
            return sent_list

    def word_tokenize(self, text):
            """
            Tokenizes the input text into individual words.

            Args:
                text (str): The text to tokenize.

            Returns:
                list: A list of individual words in the input text.
            """
            return [w for sent in nltk.sent_tokenize(text) for w in nltk.word_tokenize(sent)]

    def remove_stopwords(self,sentence):
            """
            Removes stopwords from a given sentence.

            Args:
                sentence (str): The sentence to remove stopwords from.

            Returns:
                str: The sentence with stopwords removed.
            """
            stop_words = stopwords.words('english')
            return ' '.join([w for w in nltk.word_tokenize(sentence) if not w in stop_words])

    def stem(self, text):
            """
            Stem the input text using the Snowball Stemmer.

            Args:
                text (str): The input text to be stemmed.

            Returns:
                str: The stemmed text.
            """
            stemmed_word = [snowball_stemmer.stem(word) for sent in nltk.sent_tokenize(text)for word in nltk.word_tokenize(sent)]
            return " ".join(stemmed_word)

    def lemmatize(self, text):
            """
            Lemmatizes the input text using the WordNet lemmatizer.

            Args:
                text (str): The text to lemmatize.

            Returns:
                str: The lemmatized text.
            """
            lemmatized_word = [wordnet_lemmatizer.lemmatize(word)for sent in nltk.sent_tokenize(text)for word in nltk.word_tokenize(sent)]
            return " ".join(lemmatized_word)


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