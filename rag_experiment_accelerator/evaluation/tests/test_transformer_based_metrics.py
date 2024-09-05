from unittest.mock import MagicMock

import numpy as np


from rag_experiment_accelerator.evaluation.transformer_based_metrics import (
    compare_semantic_document_values,
)


def test_compare_semantic_document_values():
    mock_sentence_transformer = MagicMock()
    embeddings1 = np.array([[0.1, 0.2, 0.3, 0.4, 0.7]])
    embeddings2 = np.array([[0.1, 0.3, 0.4, 0.5, 0.6]])

    mock_sentence_transformer.encode.side_effect = [embeddings1, embeddings2]

    value1 = "value1"
    value2 = "value2"

    assert (
        compare_semantic_document_values(value1, value2, mock_sentence_transformer)
        == 97
    )
