from rag_experiment_accelerator.evaluation.spacy_evaluator import SpacyEvaluator


# TODO: Test model download as well
def test_evaluator_init():
    similarity_threshold = 0.4
    evaluator = SpacyEvaluator(similarity_threshold=similarity_threshold)
    assert similarity_threshold == evaluator.similarity_threshold


def test_similarity_returns_similar():
    evaluator = SpacyEvaluator()
    assert evaluator.similarity("test word", "test word") == 1


def test_is_relevant_returns_valid():
    evaluator = SpacyEvaluator()
    assert evaluator.is_relevant("test phrase", "test phrase") is True
    assert evaluator.is_relevant("phrase", "different") is False
