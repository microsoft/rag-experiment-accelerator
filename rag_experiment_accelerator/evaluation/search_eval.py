from sklearn import metrics

from rag_experiment_accelerator.evaluation.spacy_evaluator import (
    SpacyEvaluator,
)
from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


def evaluate_search_result(
    search_response: list, evaluation_content: str, evaluator: SpacyEvaluator
):
    content = []

    # create list of all docs with their is_relevant result to calculate recall and precision
    is_relevant_results = []
    for doc in search_response:
        is_relevant = evaluator.is_relevant(doc["content"], evaluation_content)
        is_relevant_results.append(is_relevant)

    recall_scores = []
    precision_scores = []
    recall_predictions = [False for _ in range(len(search_response))]
    precision_predictions = [True for _ in range(len(search_response))]
    for i, doc in enumerate(search_response):
        k = i + 1
        logger.info("++++++++++++++++++++++++++++++++++")
        logger.info(f"Content: {doc['content']}")
        logger.info(f"Search Score: {doc['@search.score']}")

        precision_score = round(
            metrics.precision_score(
                is_relevant_results[:k], precision_predictions[:k]
            ),
            2,
        )
        precision_scores.append(precision_score)
        logger.info(f"Precision Score: {precision_score}@{k}")

        recall_predictions[i] = is_relevant_results[i]
        recall_score = round(
            metrics.recall_score(is_relevant_results, recall_predictions), 2
        )
        recall_scores.append(recall_score)
        logger.info(f"Recall Score: {recall_score}@{k}")

        # TODO: should we only append content when it is relevant?
        content.append(doc["content"])

    eval_metrics = {
        "recall_scores": recall_scores,
        "precision_scores": precision_scores,
    }

    return content, eval_metrics
