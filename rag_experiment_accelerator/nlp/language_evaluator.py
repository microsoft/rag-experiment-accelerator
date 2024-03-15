from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.config.environment import Environment

logger = get_logger(__name__)


class LanguageEvaluator:
    """
    A class for detecting language on text using the built-in Language Detection skill in Azure AI Services.

    Args:
        query_language: The language of the query. Possible values include: "none", "en-us",
         "en-gb", "en-in", "en-ca", "en-au", "fr-fr", "fr-ca", "de-de", "es-es", "es-mx", "zh-cn",
         "zh-tw", "pt-br", "pt-pt", "it-it", "ja-jp", "ko-kr", "ru-ru", "cs-cz", "nl-be", "nl-nl",
         "hu-hu", "pl-pl", "sv-se", "tr-tr", "hi-in", "ar-sa", "ar-eg", "ar-ma", "ar-kw", "ar-jo",
         "da-dk", "no-no", "bg-bg", "hr-hr", "hr-ba", "ms-my", "ms-bn", "sl-sl", "ta-in", "vi-vn",
         "el-gr", "ro-ro", "is-is", "id-id", "th-th", "lt-lt", "uk-ua", "lv-lv", "et-ee", "ca-es",
         "fi-fi", "sr-ba", "sr-me", "sr-rs", "sk-sk", "nb-no", "hy-am", "bn-in", "eu-es", "gl-es",
         "gu-in", "he-il", "ga-ie", "kn-in", "ml-in", "mr-in", "fa-ae", "pa-in", "te-in", "ur-pk".
        default_language: The ISO 6391 language code for the language identified. For example, "en".
        country_hint (str): An ISO 3166-1 alpha-2 two letter country code to use as a hint to the language detection model if it cannot disambiguate the language.
        confidence_threshold (float): The minimum confidence score required for language detected to be considered reliable.

    Attributes:
        query_language: The language of the query
        default_language: The ISO 6391 language code for the language identified. For example, "en".
        country_hint (str): An ISO 3166-1 alpha-2 two letter country code to use as a hint to the language detection model if it cannot disambiguate the language.
        confidence_threshold (float): The minimum confidence score required for two documents to be considered relevant.
        max_content_length (int): The maximum size of a content allowed measured by length (e.g. 50,000 characters)

    Methods:
        detect_language(text: str | list[str]) -> Dict[str, str] | None: Detect language for a text sample or a batch of documents.
        is_confident(text: str) -> bool: Determines whether language detected is reliable based on confidence score.
        is_language_match(text: str, language_code: str) -> bool: Determines whether language matches language detected.
        check_string(text: str) -> Check the length of an input string.
    """

    def __init__(
        self,
        environment: Environment,
        query_language="en-us",
        default_language="en",
        country_hint="",
        confidence_threshold=0.8,
    ) -> None:
        try:
            self.query_language = query_language
            self.default_language = (
                default_language if default_language else query_language.split("-")[0]
            )
            self.country_hint = (
                country_hint if country_hint else query_language.split("-")[1]
            )
            self.confidence_threshold = confidence_threshold
            self.max_content_length = 50000  # Data limit
            self.environment = environment
        except Exception as e:
            logger.error(str(e))

    def check_string(self, input_string):
        try:
            if not isinstance(input_string, str):
                raise ValueError("Input must be a string")
            if len(input_string) < self.max_content_length:
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return None

    def detect_language(self, text: str):
        try:
            client = TextAnalyticsClient(
                endpoint=self.environment.azure_language_service_endpoint,
                credential=AzureKeyCredential(
                    self.environment.azure_language_service_key
                ),
            )
            response = client.detect_language(documents=[text])

            for doc in response:
                if not doc.is_error:
                    logger.info(f"Detected language: {doc.primary_language}")
                else:
                    logger.error(f"Unable to detect language: {doc.id} {doc.error}")
            client.close()
            return {
                "name": doc.primary_language.name,
                "confidence_score": doc.primary_language.confidence_score,
                "iso6391_name": doc.primary_language.iso6391_name,
            }
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return None

    def is_confident(self, text: str):
        primary_language = self.detect_language(text)
        confidence_score = primary_language.get("confidence_score")
        language = primary_language.get("name")
        logger.info(f"Language: {language} Confidence Score: {confidence_score}")

        return confidence_score >= self.confidence_threshold

    def is_language_match(self, text: str, language_code: str):
        primary_language = self.detect_language(text)
        confidence_score = primary_language.get("confidence_score")
        language = primary_language.get("name")
        logger.info(f"Language: {language} Confidence Score: {confidence_score}")

        return (
            language_code == primary_language.get("iso6391_name")
            and confidence_score >= self.confidence_threshold
        )
