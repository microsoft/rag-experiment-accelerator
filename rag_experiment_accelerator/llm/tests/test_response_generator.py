import unittest
from unittest.mock import patch, Mock, MagicMock
from rag_experiment_accelerator.llm.exceptions import ContentFilteredException
from rag_experiment_accelerator.llm.response_generator import ResponseGenerator


class TestResponseGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = ResponseGenerator.__new__(ResponseGenerator)
        self.generator.config = Mock()
        self.generator.temperature = 0.5
        self.generator.deployment_name = "deployment_name"
        self.generator.client = Mock()

    @patch(
        "rag_experiment_accelerator.llm.response_generator.ResponseGenerator._create_chat_completion_with_retry"
    )
    def test_generate_response(self, mock_create_chat_completion_with_retry):
        # Arrange
        sys_message = "system message"
        prompt = "user prompt"
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "mock content"
        mock_create_chat_completion_with_retry.return_value = mock_response

        # Act
        result = self.generator.generate_response(sys_message, prompt)

        # Assert
        self.assertEqual(result, "mock content")
        mock_create_chat_completion_with_retry.assert_called_once()

        # Assert on None
        mock_create_chat_completion_with_retry.reset_mock()
        mock_response.choices[0].message.content = None

        mock_create_chat_completion_with_retry.return_value = mock_response
        result = self.generator.generate_response(sys_message, prompt)
        self.assertEqual(result, None)

        mock_create_chat_completion_with_retry.assert_called_once()

        # Assert on content_filter
        mock_create_chat_completion_with_retry.reset_mock()
        mock_response.choices[0].finish_reason = "content_filter"
        mock_create_chat_completion_with_retry.return_value = mock_response
        with self.assertRaises(ContentFilteredException) as context:
            result = self.generator.generate_response(sys_message, prompt)
        self.assertTrue("Content was filtered." in str(context.exception))
        mock_create_chat_completion_with_retry.assert_called_once()

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
