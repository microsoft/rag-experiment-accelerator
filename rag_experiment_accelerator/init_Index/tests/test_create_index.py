import unittest
from unittest.mock import patch, Mock

from rag_experiment_accelerator.init_Index.create_index import create_acs_index

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswParameters,
    HnswVectorSearchAlgorithmConfiguration,
    SearchField,
    SearchIndex,
)


class TestCreateIndex(unittest.TestCase):
    # Create a mock object for the result of the 'create_or_update_index'
    # method
    mock_result = Mock()
    mock_result.name = "test_index"

    # Test that create_acs_index calls create_or_update_index using key
    # credentials
    @patch.object(AzureKeyCredential, "__init__", return_value=None)
    @patch.object(SearchIndexClient, "create_or_update_index", return_value=mock_result)
    def test_create_acs_index(
        self, mock_create_or_update_index, mock_azure_key_credential
    ):
        # Index test parameters
        service_endpoint = "test_endpoint"
        index_name = "test_index"
        key = "test_key"
        dimension = 128
        ef_construction = 100
        ef_search = 100
        analyzers = {
            "index_analyzer_name": None,
            "search_analyzer_name": None,
            "analyzer_name": None,
            "tokenizers": None,
            "token_filters": None,
            "char_filters": None,
        }

        # Call the function with the test parameters
        create_acs_index(
            service_endpoint,
            index_name,
            key,
            dimension,
            ef_construction,
            ef_search,
            analyzers,
        )

        # Assert that the 'create_or_update_index' method was called on the
        # SearchIndexClient
        mock_create_or_update_index.assert_called_once()

        # Assert that the AzureKeyCredential was initialized with the correct
        # key
        mock_azure_key_credential.assert_called_once_with(key)

    # Test that create_acs_index works correctly when analyzer option is set
    # alone
    @patch.object(AzureKeyCredential, "__init__", return_value=None)
    @patch.object(SearchField, "__init__", return_value=None)
    @patch.object(SearchIndexClient, "create_or_update_index", return_value=mock_result)
    def test_analyzer_name_alone(
        self, mock_create_or_update_index, mock_seearch_field, mock_azure_key_credential
    ):
        service_endpoint = "test_endpoint"
        index_name = "test_index"
        key = "test_key"
        dimension = 128
        ef_construction = 100
        ef_search = 100
        analyzers = {
            "analyzer_name": "test_analyzer",
            "index_analyzer_name": None,
            "search_analyzer_name": None,
        }
        create_acs_index(
            service_endpoint,
            index_name,
            key,
            dimension,
            ef_construction,
            ef_search,
            analyzers,
        )
        mock_create_or_update_index.assert_called()

    # Test that create_acs_index works correctly when indexAnalyzer is set
    # together with searchAnalyzer and not analyzer option
    def test_analyzer_with_index_and_search_analyzer(self):
        with self.assertRaises(ValueError):
            service_endpoint = "test_endpoint"
            index_name = "test_index"
            key = "test_key"
            dimension = 128
            ef_construction = 100
            ef_search = 100
            analyzers = {
                "analyzer_name": None,
                "index_analyzer_name": "test_index_analyzer",
                "search_analyzer_name": "test_search_analyzer",
            }
            create_acs_index(
                service_endpoint,
                index_name,
                key,
                dimension,
                ef_construction,
                ef_search,
                analyzers,
            )

            # Test if only one of index_analyzer_name or search_analyzer_name is set.
            analyzers = {
                "analyzer_name": None,
                "index_analyzer_name": None,
                "search_analyzer_name": "test_search_analyzer",
            }
            create_acs_index(
                service_endpoint,
                index_name,
                key,
                dimension,
                ef_construction,
                ef_search,
                analyzers,
            )
            self.assertRaises(
                Exception,
                create_acs_index,
                service_endpoint,
                index_name,
                key,
                dimension,
                ef_construction,
                ef_search,
                analyzers,
            )

            analyzers = {
                "analyzer_name": None,
                "index_analyzer_name": "test_index_analyzer",
                "search_analyzer_name": None,
            }
            create_acs_index(
                service_endpoint,
                index_name,
                key,
                dimension,
                ef_construction,
                ef_search,
                analyzers,
            )
            self.assertRaises(
                Exception,
                create_acs_index,
                service_endpoint,
                index_name,
                key,
                dimension,
                ef_construction,
                ef_search,
                analyzers,
            )

    # Test that create_acs_index raiser error when analyzer is set together
    # with either searchAnalyzer or indexAnalyzer

    @patch.object(AzureKeyCredential, "__init__", return_value=None)
    @patch.object(SearchIndexClient, "create_or_update_index", return_value=mock_result)
    def test_analyzer_with_index_or_search_analyzer(
        self, mock_create_or_update_index, mock_azure_key_credential
    ):
        with self.assertRaises(ValueError):
            service_endpoint = "test_endpoint"
            index_name = "test_index"
            key = "test_key"
            dimension = 128
            ef_construction = 100
            ef_search = 100
            analyzers = {
                "analyzer_name": "test_analyzer",
                "index_analyzer_name": "test_index_analyzer",
                "search_analyzer_name": None,
            }
            create_acs_index(
                service_endpoint,
                index_name,
                key,
                dimension,
                ef_construction,
                ef_search,
                analyzers,
            )
            self.assertRaises(
                Exception,
                create_acs_index,
                service_endpoint,
                index_name,
                key,
                dimension,
                ef_construction,
                ef_search,
                analyzers,
            )

            analyzers = {
                "analyzer_name": "test_analyzer",
                "index_analyzer_name": None,
                "search_analyzer_name": "test_search_analyzer",
            }
            create_acs_index(
                service_endpoint,
                index_name,
                key,
                dimension,
                ef_construction,
                ef_search,
                analyzers,
            )
            self.assertRaises(
                Exception,
                create_acs_index,
                service_endpoint,
                index_name,
                key,
                dimension,
                ef_construction,
                ef_search,
                analyzers,
            )

    # Test that create_acs_index works correctly when the analyzers dictionary
    # contains non-None values
    @patch.object(AzureKeyCredential, "__init__", return_value=None)
    @patch.object(SearchIndexClient, "create_or_update_index", return_value=mock_result)
    def test_create_acs_index_analyzers_non_none(
        self, mock_create_or_update_index, mock_azure_key_credential
    ):
        analyzers = {
            "index_analyzer_name": "test_index_analyzer",
            "search_analyzer_name": "test_search_analyzer",
            "analyzer_name": None,
            "tokenizers": [
                {"name": "my_tokenizer", "token_chars": ["letter", "digit"]}
            ],
            "token_filters": ["token_filter1", "token_filter2"],
            "char_filters": [
                {"name": "my_char_filter", "mappings": ["ph=>f", "qu=>q"]}
            ],
        }
        try:
            create_acs_index(
                "test_endpoint", "test_index", "test_key", 128, 100, 100, analyzers
            )
        except Exception:
            self.fail("create_acs_index raised Exception unexpectedly!")

    # Test that create_acs_index works correctly when the analyzers dictionary
    # contains None values
    @patch.object(AzureKeyCredential, "__init__", return_value=None)
    @patch.object(SearchIndexClient, "create_or_update_index", return_value=mock_result)
    def test_create_acs_index_analyzers_none(
        self, mock_create_or_update_index, mock_azure_key_credential
    ):
        analyzers = {
            "index_analyzer_name": None,
            "search_analyzer_name": None,
            "analyzer_name": None,
            "tokenizers": None,
            "token_filters": None,
            "char_filters": None,
        }
        try:
            create_acs_index(
                "test_endpoint", "test_index", "test_key", 128, 100, 100, analyzers
            )
        except Exception:
            self.fail("create_acs_index raised Exception unexpectedly!")

    # Test that create_acs_index raises an exception when given invalid
    # parameters:
    @patch.object(AzureKeyCredential, "__init__", return_value=None)
    @patch.object(SearchIndexClient, "create_or_update_index", return_value=mock_result)
    def test_create_acs_index_invalid_parameters(
        self, mock_create_or_update_index, mock_azure_key_credential
    ):
        with self.assertRaises(ValueError):
            create_acs_index(None, "test_index", "test_key", 128, 100, 100, {})

    # Test that create_acs_index raises an exception when the
    # create_or_update_index method fails
    @patch.object(AzureKeyCredential, "__init__", return_value=None)
    @patch.object(SearchIndexClient, "create_or_update_index", side_effect=Exception)
    def test_create_acs_index_create_or_update_index_fails(
        self, mock_create_or_update_index, mock_azure_key_credential
    ):
        with self.assertRaises(Exception):
            create_acs_index(
                "test_endpoint", "test_index", "test_key", 128, 100, 100, {}
            )

    # Test that create_acs_index works correctly when the
    # create_or_update_index method returns a non-None value
    @patch.object(AzureKeyCredential, "__init__", return_value=None)
    @patch.object(SearchIndexClient, "create_or_update_index", return_value=mock_result)
    def test_create_acs_index_create_or_update_index_returns_non_none(
        self, mock_create_or_update_index, mock_azure_key_credential
    ):
        try:
            create_acs_index(
                "test_endpoint", "test_index", "test_key", 128, 100, 100, {}
            )
        except Exception:
            self.fail("create_acs_index raised Exception unexpectedly!")

    # Test that create_acs_index calls create_or_update_index with the correct
    # parameters
    @patch.object(AzureKeyCredential, "__init__", return_value=None)
    @patch.object(HnswParameters, "__init__", return_value=None)
    @patch.object(HnswVectorSearchAlgorithmConfiguration, "__init__", return_value=None)
    @patch.object(SearchIndex, "__init__", return_value=None)
    @patch.object(SearchField, "__init__", return_value=None)
    @patch.object(SearchIndexClient, "create_or_update_index", return_value=mock_result)
    def test_dimension_setting(
        self,
        mock_create_or_update_index,
        mock_search_field,
        mock_search_index,
        mock_hnsw_vector_search_algorithm_configuration,
        mock_hnsw_parameters,
        mock_azure_key_credential,
    ):
        # Test parameters
        service_endpoint = "test_endpoint"
        index_name = "test_index"
        key = "test_key"
        dimension = 128
        ef_construction = 100
        ef_search = 100
        analyzers = {
            "index_analyzer_name": None,
            "search_analyzer_name": None,
            "analyzer_name": None,
            "tokenizers": None,
            "token_filters": None,
            "char_filters": None,
        }
        # Call the function with the test parameters
        create_acs_index(
            service_endpoint,
            index_name,
            key,
            dimension,
            ef_construction,
            ef_search,
            analyzers,
        )
        # Assert that the 'create_or_update_index' method was called with the
        # correct dimension
        args, kwargs = mock_create_or_update_index.call_args
        searchable_fields = mock_search_field.call_args_list
        index_parameters = mock_search_index.call_args_list
        vector_search_dimensions = None
        index_name_parameter = None
        expected_dimension = dimension
        expected_index_name = index_name
        for call in searchable_fields:
            if call.kwargs["name"] == "contentVector":
                vector_search_dimensions = call
                break

        for call in index_parameters:
            if "name" in call.kwargs:
                index_name_parameter = call.kwargs["name"]
                break

        self.assertIsNotNone(index_name_parameter)
        self.assertIsNotNone(vector_search_dimensions)
        self.assertEqual(expected_index_name, index_name_parameter)
        self.assertEqual(
            expected_dimension,
            vector_search_dimensions.kwargs.get("vector_search_dimensions"),
        )
        mock_hnsw_parameters.assert_called_with(
            m=4, ef_construction=ef_construction, ef_search=ef_search, metric="cosine"
        )


if __name__ == "__main__":
    unittest.main()
