import os
import spacy
from spacy import cli

import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logging_level = os.getenv("LOGGING_LEVEL", "INFO").upper()
logger = logging.getLogger(__name__)
logger.setLevel(logging_level)  # Set level

class SpacyEvaluator():
    """
    A class for evaluating the similarity between two documents using spaCy.

    Args:
        similarity_threshold (float): The minimum similarity score required for two documents to be considered relevant.
        model (str): The name of the spaCy model to use for processing the documents.

    Attributes:
        nlp (spacy.Language): The spaCy language model used for processing the documents.
        similarity_threshold (float): The minimum similarity score required for two documents to be considered relevant.

    Methods:
        similarity(doc1: str, doc2: str) -> float: Calculates the similarity score between two documents.
        is_relevant(doc1: str, doc2: str) -> bool: Determines whether two documents are relevant based on their similarity score.
    """

    def __init__(self, similarity_threshold=0.8, model='en_core_web_md') -> None:
        cli.download(model)
        self.nlp = spacy.load(model)
        self.similarity_threshold = similarity_threshold
    
    def similarity(self, doc1: str, doc2: str):
        nlp_doc1 = self.nlp(doc1)
        nlp_doc2 = self.nlp(doc2)
        return nlp_doc1.similarity(nlp_doc2)

    def is_relevant(self, doc1: str, doc2: str):
        similarity = self.similarity(doc1, doc2)
        logger.info(f"Similarity Score: {similarity}")

        return similarity > self.similarity_threshold
