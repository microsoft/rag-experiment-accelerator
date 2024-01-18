from unittest.mock import MagicMock, patch, call
from azure.search.documents.models import (
    QueryAnswerType,
    QueryCaptionType,
    QueryLanguage,
    QueryType,
)

from rag_experiment_accelerator.search_type.acs_search_methods import (
    format_results,
    search_for_manual_hybrid,
    search_for_match_Hybrid_cross,
    search_for_match_Hybrid_multi,
    search_for_match_pure_vector,
    search_for_match_pure_vector_cross,
    search_for_match_pure_vector_multi,
    search_for_match_semantic,
    search_for_match_text,
)


def test_format_results():
    search_results = [
        {"content": "some content 1", "@search.score": 1.1},
        {"content": "some content 2", "@search.score": 1.2},
    ]
    formatted_results = format_results(search_results)

    assert formatted_results[0]["content"] == "some content 1"
    assert formatted_results[0]["@search.score"] == 1.1
    assert formatted_results[1]["content"] == "some content 2"
    assert formatted_results[1]["@search.score"] == 1.2


@patch("rag_experiment_accelerator.search_type.acs_search_methods.RawVectorQuery")
def test_search_for_match_semantic(mock_vector_query):
    # Arrange
    query = "some query"
    retrieve_num_of_documents = 5

    client = MagicMock()
    client.search.return_value = [
        {
            "content": "some content",
            "title": "some title",
            "summary": "A list of items with titles and content.",
            "@search.score": 1.1,
            "@search.reranker_score": None,
            "@search.highlights": None,
            "@search.captions": None,
        }
    ]

    embedding_model = MagicMock()
    embedding = [1, 2, 3]
    embedding_model.generate_embedding.return_value = embedding

    vector1 = "vector1"
    vector2 = "vector2"
    mock_vector_query.side_effect = [vector1, vector2]

    # Act
    results = search_for_match_semantic(
        client=client,
        embedding_model=embedding_model,
        query=query,
        retrieve_num_of_documents=retrieve_num_of_documents,
    )

    # Assert
    assert mock_vector_query.call_count == 2
    mock_vector_query.assert_has_calls(
        [
            call(k=retrieve_num_of_documents, fields="contentVector", vector=embedding),
            call(
                k=retrieve_num_of_documents,
                fields="contentTitle, contentSummary",
                vector=embedding,
            ),
        ]
    )
    client.search.assert_called_once_with(
        search_text=query,
        vector_queries=[vector1, vector2],
        top=retrieve_num_of_documents,
        select=["title", "content", "summary"],
        query_type=QueryType.SEMANTIC,
        query_language=QueryLanguage.EN_US,
        semantic_configuration_name="my-semantic-config",
        query_caption=QueryCaptionType.EXTRACTIVE,
        query_answer=QueryAnswerType.EXTRACTIVE,
    )

    assert results == [{"content": "some content", "@search.score": 1.1}]


@patch("rag_experiment_accelerator.search_type.acs_search_methods.RawVectorQuery")
def test_search_for_match_Hybrid_multi(mock_vector_query):
    query = "some query"
    retrieve_num_of_documents = 5

    client = MagicMock()
    client.search.return_value = [
        {
            "content": "some content",
            "title": "some title",
            "summary": "A list of items with titles and content.",
            "@search.score": 1.1,
            "@search.reranker_score": None,
            "@search.highlights": None,
            "@search.captions": None,
        }
    ]

    embedding_model = MagicMock()
    embedding = [1, 2, 3]
    embedding_model.generate_embedding.return_value = embedding

    vector1 = "vector1"
    vector2 = "vector2"
    vector3 = "vector3"
    mock_vector_query.side_effect = [vector1, vector2, vector3]

    # Act
    results = search_for_match_Hybrid_multi(
        client=client,
        embedding_model=embedding_model,
        query=query,
        retrieve_num_of_documents=retrieve_num_of_documents,
    )

    # Assert
    assert mock_vector_query.call_count == 3
    mock_vector_query.assert_has_calls(
        [
            call(k=retrieve_num_of_documents, fields="contentVector", vector=embedding),
            call(
                k=retrieve_num_of_documents,
                fields="contentTitle",
                vector=embedding,
            ),
            call(
                k=retrieve_num_of_documents,
                fields="contentSummary",
                vector=embedding,
            ),
        ]
    )
    client.search.assert_called_once_with(
        search_text=query,
        vector_queries=[vector1, vector2, vector3],
        top=retrieve_num_of_documents,
        select=["title", "content", "summary"],
    )

    assert results == [{"content": "some content", "@search.score": 1.1}]


@patch("rag_experiment_accelerator.search_type.acs_search_methods.RawVectorQuery")
def test_search_for_match_Hybrid_cross(mock_vector_query):
    # Arrange
    query = "some query"
    retrieve_num_of_documents = 5

    client = MagicMock()
    client.search.return_value = [
        {
            "content": "some content",
            "title": "some title",
            "summary": "A list of items with titles and content.",
            "@search.score": 1.1,
            "@search.reranker_score": None,
            "@search.highlights": None,
            "@search.captions": None,
        }
    ]

    embedding_model = MagicMock()
    embedding = [1, 2, 3]
    embedding_model.generate_embedding.return_value = embedding

    vector1 = "vector1"
    vector2 = "vector2"
    mock_vector_query.side_effect = [vector1, vector2]

    # Act
    results = search_for_match_Hybrid_cross(
        client=client,
        embedding_model=embedding_model,
        query=query,
        retrieve_num_of_documents=retrieve_num_of_documents,
    )

    # Assert
    assert mock_vector_query.call_count == 2
    mock_vector_query.assert_has_calls(
        [
            call(k=retrieve_num_of_documents, fields="contentVector", vector=embedding),
            call(
                k=retrieve_num_of_documents,
                fields="contentTitle, contentSummary",
                vector=embedding,
            ),
        ]
    )
    client.search.assert_called_once_with(
        search_text=query,
        vector_queries=[vector1, vector2],
        top=retrieve_num_of_documents,
        select=["title", "content", "summary"],
    )

    assert results == [{"content": "some content", "@search.score": 1.1}]


def test_search_for_match_text():
    # Arrange
    query = "some query"
    retrieve_num_of_documents = 5

    client = MagicMock()
    client.search.return_value = [
        {
            "content": "some content",
            "title": "some title",
            "summary": "A list of items with titles and content.",
            "@search.score": 1.1,
            "@search.reranker_score": None,
            "@search.highlights": None,
            "@search.captions": None,
        }
    ]

    # Act
    results = search_for_match_text(
        client=client,
        query=query,
        retrieve_num_of_documents=retrieve_num_of_documents,
    )

    # Assert
    client.search.assert_called_once_with(
        search_text=query,
        top=retrieve_num_of_documents,
        select=["title", "content", "summary"],
    )

    assert results == [{"content": "some content", "@search.score": 1.1}]


@patch("rag_experiment_accelerator.search_type.acs_search_methods.RawVectorQuery")
def test_search_for_match_pure_vector(mock_vector_query):
    # Arrange
    query = "some query"
    retrieve_num_of_documents = 5

    client = MagicMock()
    client.search.return_value = [
        {
            "content": "some content",
            "title": "some title",
            "summary": "A list of items with titles and content.",
            "@search.score": 1.1,
            "@search.reranker_score": None,
            "@search.highlights": None,
            "@search.captions": None,
        }
    ]

    embedding_model = MagicMock()
    embedding = [1, 2, 3]
    embedding_model.generate_embedding.return_value = embedding

    vector1 = "vector1"
    mock_vector_query.side_effect = [vector1]

    # Act
    results = search_for_match_pure_vector(
        client=client,
        embedding_model=embedding_model,
        query=query,
        retrieve_num_of_documents=retrieve_num_of_documents,
    )

    # Assert
    assert mock_vector_query.call_count == 1
    mock_vector_query.assert_has_calls(
        [
            call(k=retrieve_num_of_documents, fields="contentVector", vector=embedding),
        ]
    )
    client.search.assert_called_once_with(
        search_text=None,
        vector_queries=[vector1],
        top=retrieve_num_of_documents,
        select=["title", "content", "summary"],
    )

    assert results == [{"content": "some content", "@search.score": 1.1}]


@patch("rag_experiment_accelerator.search_type.acs_search_methods.RawVectorQuery")
def test_search_for_match_pure_vector_multi(mock_vector_query):
    # Arrange
    query = "some query"
    retrieve_num_of_documents = 5

    client = MagicMock()
    client.search.return_value = [
        {
            "content": "some content",
            "title": "some title",
            "summary": "A list of items with titles and content.",
            "@search.score": 1.1,
            "@search.reranker_score": None,
            "@search.highlights": None,
            "@search.captions": None,
        }
    ]

    embedding_model = MagicMock()
    embedding = [1, 2, 3]
    embedding_model.generate_embedding.return_value = embedding

    vector1 = "vector1"
    vector2 = "vector2"
    vector3 = "vector3"
    mock_vector_query.side_effect = [vector1, vector2, vector3]

    # Act
    results = search_for_match_pure_vector_multi(
        client=client,
        embedding_model=embedding_model,
        query=query,
        retrieve_num_of_documents=retrieve_num_of_documents,
    )

    # Assert
    assert mock_vector_query.call_count == 3
    mock_vector_query.assert_has_calls(
        [
            call(k=retrieve_num_of_documents, fields="contentVector", vector=embedding),
            call(k=retrieve_num_of_documents, fields="contentTitle", vector=embedding),
            call(
                k=retrieve_num_of_documents, fields="contentSummary", vector=embedding
            ),
        ]
    )
    client.search.assert_called_once_with(
        search_text=None,
        vector_queries=[vector1, vector2, vector3],
        top=retrieve_num_of_documents,
        select=["title", "content", "summary"],
    )

    assert results == [{"content": "some content", "@search.score": 1.1}]


@patch("rag_experiment_accelerator.search_type.acs_search_methods.RawVectorQuery")
def test_search_for_match_pure_vector_cross(mock_vector_query):
    # Arrange
    query = "some query"
    retrieve_num_of_documents = 5

    client = MagicMock()
    client.search.return_value = [
        {
            "content": "some content",
            "title": "some title",
            "summary": "A list of items with titles and content.",
            "@search.score": 1.1,
            "@search.reranker_score": None,
            "@search.highlights": None,
            "@search.captions": None,
        }
    ]

    embedding_model = MagicMock()
    embedding = [1, 2, 3]
    embedding_model.generate_embedding.return_value = embedding

    vector1 = "vector1"
    mock_vector_query.return_value = vector1

    # Act
    results = search_for_match_pure_vector_cross(
        client=client,
        embedding_model=embedding_model,
        query=query,
        retrieve_num_of_documents=retrieve_num_of_documents,
    )

    # Assert
    assert mock_vector_query.call_count == 1
    mock_vector_query.assert_has_calls(
        [
            call(
                k=retrieve_num_of_documents,
                fields="contentVector, contentTitle, contentSummary",
                vector=embedding,
            )
        ]
    )
    client.search.assert_called_once_with(
        search_text=None,
        vector_queries=[vector1],
        top=retrieve_num_of_documents,
        select=["title", "content", "summary"],
    )

    assert results == [{"content": "some content", "@search.score": 1.1}]


@patch(
    "rag_experiment_accelerator.search_type.acs_search_methods.search_for_match_text"
)
@patch(
    "rag_experiment_accelerator.search_type.acs_search_methods.search_for_match_pure_vector_cross"
)
@patch(
    "rag_experiment_accelerator.search_type.acs_search_methods.search_for_match_semantic"
)
def test_search_for_manual_hybrid(
    mock_search_for_match_semantic,
    mock_search_for_match_pure_vector_cross,
    mock_search_for_match_text,
):
    query = "some query"
    retrieve_num_of_documents = 5
    client = MagicMock()
    embedding_model = MagicMock()

    mock_search_for_match_text.return_value = [
        {"content": "some content 1", "@search.score": 1.1}
    ]
    mock_search_for_match_pure_vector_cross.return_value = [
        {"content": "some content 2", "@search.score": 1.2}
    ]
    mock_search_for_match_semantic.return_value = [
        {"content": "some content 3", "@search.score": 1.3}
    ]

    results = search_for_manual_hybrid(
        client=client,
        embedding_model=embedding_model,
        query=query,
        retrieve_num_of_documents=retrieve_num_of_documents,
    )

    assert results == [
        {"content": "some content 1", "@search.score": 1.1},
        {"content": "some content 2", "@search.score": 1.2},
        {"content": "some content 3", "@search.score": 1.3},
    ]