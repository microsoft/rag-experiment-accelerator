import unittest
import json
from unittest.mock import patch, Mock
from rag_experiment_accelerator.llm.exceptions import ContentFilteredException
from rag_experiment_accelerator.llm.response_generator import ResponseGenerator
from rag_experiment_accelerator.llm.prompt import (
    StructuredPrompt,
    CoTPrompt,
    Prompt,
    PromptTag,
)


class TestResponseGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = ResponseGenerator.__new__(ResponseGenerator)
        self.generator.config = Mock()
        self.generator.temperature = 0.5
        self.generator.deployment_name = "deployment_name"
        self.generator.client = Mock()
        self.generator.json_object_supported = False
        self.prompt = Mock(spec=Prompt)
        self.prompt.tags = {}

    def create_mock_prompt(self, prompt_type, tags, separator=None, validator=None):
        mock_prompt = Mock(spec=prompt_type)
        mock_prompt.tags = tags
        if separator:
            mock_prompt.separator = separator
        if validator:
            mock_prompt.validator = validator
        return mock_prompt

    def test_interpret_response_with_cot_prompt(self):
        response = "Introduction##RESPONSE##Detailed explanation"
        prompt = self.create_mock_prompt(
            CoTPrompt, [PromptTag.ChainOfThought], separator="##RESPONSE##"
        )
        result = self.generator._interpret_response(response, prompt)
        self.assertEqual(result, "Detailed explanation")

    def test_interpret_response_with_structured_prompt(self):
        response = '{"key": "value"}'
        prompt = self.create_mock_prompt(
            StructuredPrompt,
            [PromptTag.Structured, PromptTag.JSON],
            validator=lambda x: "key" in x,
        )
        result = self.generator._interpret_response(response, prompt)
        expected = json.loads(response)
        self.assertEqual(result, expected)

    def test_interpret_response_with_invalid_separator(self):
        response = "No separator present here"
        prompt = self.create_mock_prompt(
            CoTPrompt, [PromptTag.ChainOfThought], separator="##RESPONSE##"
        )
        with self.assertRaises(AssertionError):
            self.generator._interpret_response(response, prompt)

    def test_interpret_response_non_strict_mode(self):
        response = "Some response"
        prompt = self.create_mock_prompt(Prompt, [PromptTag.NonStrict])
        result = self.generator._interpret_response(response, prompt)
        self.assertEqual(result, response)

    @patch("rag_experiment_accelerator.llm.response_generator.logger")
    def test_get_response_normal(self, mock_logger):
        # Mocking the API response
        responses = [
            Mock(message=Mock(content="test response"), finish_reason="completed"),
        ]
        mock_response = Mock()
        mock_response.choices = responses
        self.generator.client.chat.completions.create.return_value = mock_response

        # Test
        result = self.generator._get_response("message", self.prompt)
        self.assertEqual(result, "test response")
        self.generator.client.chat.completions.create.assert_called_once()

    @patch("rag_experiment_accelerator.llm.response_generator.logger")
    def test_get_response_content_filtered(self, mock_logger):
        # Mocking the API response for content filtering
        mock_response = Mock()
        mock_response.choices = [
            Mock(message=Mock(content=None), finish_reason="content_filter")
        ]
        self.generator.client.chat.completions.create.return_value = mock_response

        # Test and assert exception
        with self.assertRaises(ContentFilteredException):
            self.generator._get_response("message", self.prompt)

    @patch("rag_experiment_accelerator.llm.response_generator.logger")
    def test_get_response_retries_on_random_exception(self, mock_logger):
        # Simulating an exception that should trigger a retry
        self.generator.client.chat.completions.create.side_effect = [
            Exception("Simulate API failure"),
            Mock(
                choices=[
                    Mock(
                        message=Mock(content="recovered response"),
                        finish_reason="completed",
                    )
                ]
            ),
        ]

        # Test
        result = self.generator._get_response("message", self.prompt)
        self.assertEqual(result, "recovered response")
        self.assertEqual(self.generator.client.chat.completions.create.call_count, 2)

    @patch(
        "rag_experiment_accelerator.llm.response_generator.ResponseGenerator._get_response"
    )
    def test_generate_response_full_system_message(self, mock_get_response):
        # Setup
        mock_get_response.return_value = "valid response"
        prompt = Prompt("${argument_1} ${argument_2}", "", [])
        kwargs = {"argument_1": 1, "argument_2": 2}

        # Action
        response = self.generator.generate_response(prompt, None, **kwargs)

        # Assert
        mock_get_response.assert_called_once()
        self.assertEqual(response, "valid response")

    @patch(
        "rag_experiment_accelerator.llm.response_generator.ResponseGenerator._get_response"
    )
    def test_generate_response_full_user_template(self, mock_get_response):
        # Setup
        mock_get_response.return_value = "valid response"
        prompt = Prompt("", "${argument_1} ${argument_2}", [])
        kwargs = {"argument_1": 1, "argument_2": 2}

        # Action
        response = self.generator.generate_response(prompt, None, **kwargs)

        # Assert
        mock_get_response.assert_called_once()
        self.assertEqual(response, "valid response")

    @patch(
        "rag_experiment_accelerator.llm.response_generator.ResponseGenerator._get_response"
    )
    def test_generate_response_mixed_messages(self, mock_get_response):
        # Setup
        mock_get_response.return_value = "valid response"
        prompt = Prompt("${argument_1}", "${argument_2}", [])
        kwargs = {"argument_1": 1, "argument_2": 2}

        # Action
        response = self.generator.generate_response(prompt, None, **kwargs)

        # Assert
        mock_get_response.assert_called_once()
        self.assertEqual(response, "valid response")

    @patch(
        "rag_experiment_accelerator.llm.response_generator.ResponseGenerator._get_response"
    )
    def test_generate_response_missing_system_argument(self, mock_get_response):
        # Setup
        prompt = Prompt("${argument_1}", "${argument_2}", [])
        kwargs = {"argument_1": 1}

        # Action & Assert
        with self.assertRaises(AssertionError):
            self.generator.generate_response(prompt, None, **kwargs)

    @patch(
        "rag_experiment_accelerator.llm.response_generator.ResponseGenerator._get_response"
    )
    def test_generate_response_missing_user_argument_non_strict(
        self, mock_get_response
    ):
        # Setup
        mock_get_response.side_effect = Exception("Random failure")
        prompt = Prompt("${argument_1}", "${argument_2}", [PromptTag.NonStrict])
        kwargs = {"argument_1": 1, "argument_2": 2}

        # Action
        response = self.generator.generate_response(prompt, None, **kwargs)

        # Assert
        self.assertIsNone(response)

    @patch(
        "rag_experiment_accelerator.llm.response_generator.ResponseGenerator._get_response",
        side_effect=Exception("Random failure"),
    )
    def test_generate_response_exception_handling_strict(self, mock_get_response):
        # Setup
        prompt = Prompt("${argument_1}", "${argument_2}", [])
        kwargs = {"argument_1": 1, "argument_2": 2}

        # Action & Assert
        with self.assertRaises(Exception):
            self.generator.generate_response(prompt, None, **kwargs)

    @patch(
        "rag_experiment_accelerator.llm.response_generator.ResponseGenerator._initialize_azure_openai_client"
    )
    def test_initialize_azure_openai_client(self, mock_initialize_azure_openai_client):
        # Arrange
        mock_initialize_azure_openai_client.return_value = "mock client"

        # Act
        result = self.generator._initialize_azure_openai_client()

        # Assert
        self.assertEqual(result, "mock client")
        mock_initialize_azure_openai_client.assert_called_once()


if __name__ == "__main__":
    unittest.main()
