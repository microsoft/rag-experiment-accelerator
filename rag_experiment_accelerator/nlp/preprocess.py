import re

from abc import ABC
from string import punctuation
from typing import Optional

from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


class Preprocess(ABC):
    def __init__(self, name: Optional[str] = None):
        self.name = name if name else self.__class__.__name__

    def preprocess(self, text):
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

    def remove_punct(self, text):
        """
        Removes all punctuation from the given text and returns the result.

        Args:
            text (str): The text to remove punctuation from.

        Returns:
            str: The text with all punctuation removed.
        """
        return "".join(c for c in text if c not in punctuation)

    def remove_Tags(self, text):
        """
        Removes HTML tags from the given text.

        Args:
            text (str): The text to remove HTML tags from.

        Returns:
            str: The cleaned text with HTML tags removed.
        """
        cleaned_text = re.sub("<[^<]+?>", "", text)
        return cleaned_text
