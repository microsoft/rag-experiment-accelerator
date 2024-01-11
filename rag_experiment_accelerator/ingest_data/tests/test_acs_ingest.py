import json

from unittest.mock import patch, Mock
from langchain.docstore.document import Document

from rag_experiment_accelerator.ingest_data.acs_ingest import (
    generate_title,
    my_hash,
    generate_summary,
    upload_data,
    generate_qna,
    we_need_multiple_questions,
    do_we_need_multiple_questions,
)

from rag_experiment_accelerator.llm.prompts import (
    prompt_instruction_title,
    prompt_instruction_summary,
    multiple_prompt_instruction,
)


def test_my_hash_with_string():
    # Arrange
    test_string = "Hello, World!"
    expected_hash = (
        "65a8e27d8879283831b664bd8b7f0ad4"  # Precomputed MD5 hash of "Hello, World!"
    )

    # Act
    result = my_hash(test_string)

    # Assert
    assert result == expected_hash


def test_my_hash_with_empty_string():
    # Arrange
    test_string = ""
    expected_hash = (
        "d41d8cd98f00b204e9800998ecf8427e"  # Precomputed MD5 hash of an empty string
    )

    # Act
    result = my_hash(test_string)

    # Assert
    assert result == expected_hash


def test_my_hash_with_numbers():
    # Arrange
    test_string = "1234567890"
    expected_hash = (
        "e807f1fcf82d132f9bb018ca6738a19f"  # Precomputed MD5 hash of "1234567890"
    )

    # Act
    result = my_hash(test_string)

    # Assert
    assert result == expected_hash


@patch("rag_experiment_accelerator.ingest_data.acs_ingest.ResponseGenerator")
def test_generate_title(mock_response_generator):
    # Arrange
    mock_response = "Test Title"
    mock_chunk = "This is a test chunk of text."
    mock_deployment_name = "TestDeployment"
    mock_response_generator().generate_response.return_value = mock_response

    # Act
    result = generate_title(mock_chunk, mock_deployment_name)

    # Assert
    assert result == mock_response
    # MockResponseGenerator.assert_called_once_with(
    #     deployment_name=mock_deployment_name
    # )
    mock_response_generator().generate_response.assert_called_once_with(
        prompt_instruction_title, mock_chunk
    )


@patch("rag_experiment_accelerator.ingest_data.acs_ingest.ResponseGenerator")
def test_generate_summary(mock_response_generator):
    # Arrange
    mock_summary = "Test Summary"
    mock_chunk = "This is a test chunk of text."
    mock_deployment_name = "TestDeployment"
    mock_response_generator().generate_response.return_value = mock_summary

    # Act
    result = generate_summary(mock_chunk, mock_deployment_name)

    # Assert
    assert result == mock_summary

    mock_response_generator().generate_response.assert_called_once_with(
        prompt_instruction_summary, mock_chunk
    )


@patch("rag_experiment_accelerator.ingest_data.acs_ingest.SearchClient")
@patch("rag_experiment_accelerator.ingest_data.acs_ingest.AzureKeyCredential")
@patch("rag_experiment_accelerator.ingest_data.acs_ingest.generate_title")
@patch("rag_experiment_accelerator.ingest_data.acs_ingest.generate_summary")
@patch("rag_experiment_accelerator.ingest_data.acs_ingest.my_hash")
@patch("rag_experiment_accelerator.ingest_data.acs_ingest.pre_process.preprocess")
def test_upload_data(
    mock_preprocess,
    mock_my_hash,
    mock_generate_summary,
    mock_generate_title,
    mock_AzureKeyCredential,
    mock_SearchClient,
):
    # Arrange
    mock_chunks = [{"content": "test content", "content_vector": "test_vector"}]
    mock_service_endpoint = "test_endpoint"
    mock_index_name = "test_index"
    mock_search_key = "test_key"
    mock_embedding_model = Mock()
    mock_azure_oai_deployment_name = "test_deployment"
    mock_my_hash.return_value = "test_hash"
    mock_generate_title.return_value = "test_title"
    mock_generate_summary.return_value = "test_summary"
    mock_preprocess.return_value = "test_preprocessed_content"
    mock_AzureKeyCredential.return_value = "test_credential"

    # Act
    upload_data(
        mock_chunks,
        mock_service_endpoint,
        mock_index_name,
        mock_search_key,
        mock_embedding_model,
        mock_azure_oai_deployment_name,
    )

    # Assert
    mock_AzureKeyCredential.assert_called_once_with(mock_search_key)
    mock_SearchClient.assert_called_once_with(
        endpoint=mock_service_endpoint,
        index_name=mock_index_name,
        credential="test_credential",
    )
    mock_my_hash.assert_called_once_with(mock_chunks[0]["content"])
    mock_generate_title.assert_called_once_with(
        str(mock_chunks[0]["content"]), mock_azure_oai_deployment_name
    )
    mock_generate_summary.assert_called_once_with(
        str(mock_chunks[0]["content"]), mock_azure_oai_deployment_name
    )
    mock_preprocess.assert_any_call("test_summary")
    mock_preprocess.assert_any_call("test_title")
    mock_embedding_model.generate_embedding.assert_any_call(
        chunk="test_preprocessed_content"
    )
    assert mock_embedding_model.generate_embedding.call_count == 2
    mock_SearchClient().upload_documents.assert_called_once()


@patch("rag_experiment_accelerator.ingest_data.acs_ingest.ResponseGenerator")
def test_generate_qna(mock_response_generator):
    # Arrange
    content = "This is a test document content. This needs to be above 50 characters because we don't generate responses for chunks of less than size 50. Speaking of which, this is a very long sentence."
    mock_docs = [
        Document(page_content=content),
    ]
    mock_deployment_name = "TestDeployment"
    mock_response = '[{"role": "user", "content": "Test question?"}, {"role": "assistant", "content": "Test answer."}]'
    mock_response_generator().generate_response.return_value = mock_response

    # Act
    result = generate_qna(mock_docs, mock_deployment_name)

    # Assert
    mock_response_generator().generate_response.assert_called_once()
    assert len(result) == 1
    assert result.iloc[0]["user_prompt"] == "Test question?"
    assert result.iloc[0]["output_prompt"] == "Test answer."
    assert result.iloc[0]["context"] == content


@patch("rag_experiment_accelerator.ingest_data.acs_ingest.json.loads")
@patch("rag_experiment_accelerator.ingest_data.acs_ingest.ResponseGenerator")
def test_generate_qna_with_invalid_json(mock_response_generator, mock_json_loads):
    # Arrange
    mock_docs = [
        Mock(
            page_content="This is a test document content with extras so we reach the 50 mark for response to be called, there is NO Path for less than 50"
        )
    ]
    mock_deployment_name = "TestDeployment"
    mock_response = "Invalid JSON"
    mock_response_generator().generate_response.return_value = mock_response
    mock_json_loads.side_effect = json.JSONDecodeError("Invalid JSON", doc="", pos=0)

    # Act
    result = generate_qna(mock_docs, mock_deployment_name)

    # Assert
    assert len(result) == 0
    mock_json_loads.assert_called_once_with(mock_response)


@patch("rag_experiment_accelerator.ingest_data.acs_ingest.ResponseGenerator")
def test_we_need_multiple_questions(mock_response_generator):
    # Arrange
    question = "What is the meaning of life?"
    azure_oai_deployment_name = "TestDeployment"
    mock_response = "The meaning of life is 42."
    mock_response_generator().generate_response.return_value = mock_response
    expected_prompt_instruction = (
        multiple_prompt_instruction + "\n" + "question: " + question + "\n"
    )

    # Act
    result = we_need_multiple_questions(question, azure_oai_deployment_name)

    # Assert
    mock_response_generator.assert_called_with(
        deployment_name=azure_oai_deployment_name
    )
    mock_response_generator().generate_response.assert_called_once_with(
        expected_prompt_instruction, ""
    )
    assert result == mock_response


@patch("rag_experiment_accelerator.ingest_data.acs_ingest.ResponseGenerator")
def test_do_we_need_multiple_questions_high(mock_response_generator):
    # Arrange
    question = "What is the meaning of life?"
    azure_oai_deployment_name = "TestDeployment"
    mock_response_generator().generate_response.return_value = "HIGH"

    # Act
    result = do_we_need_multiple_questions(question, azure_oai_deployment_name)

    # Assert
    assert result is not None
    mock_response_generator().generate_response.assert_called_once()


@patch("rag_experiment_accelerator.ingest_data.acs_ingest.ResponseGenerator")
def test_do_we_need_multiple_questions_empty(mock_response_generator):
    # Arrange
    question = "What is the meaning of life?"
    azure_oai_deployment_name = "TestDeployment"
    mock_response_generator().generate_response.return_value = ""

    # Act
    result = do_we_need_multiple_questions(question, azure_oai_deployment_name)

    # Assert
    assert result is None
    mock_response_generator().generate_response.assert_called_once()
