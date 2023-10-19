from evaluation.spacy_evaluator import SpacyEvaluator
from sklearn import metrics


def evaluate_search_result(search_response: list, evaluation_content: str, evaluator: SpacyEvaluator):
    content = []

    # create list of all docs with their is_relevant result to calculate recall
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
        print("++++++++++++++++++++++++++++++++++")
        print(f"Content: {doc['content']}")
        print(f"Search Score: {doc['@search.score']}")


        precision_score = metrics.precision_score(is_relevant_results[:k], precision_predictions[:k])
        precision_scores.append(f"{precision_score}@{k}")
        print(f"Precision Score: {precision_score}@{k}")

        recall_predictions[i] = is_relevant_results[i]
        recall_score = metrics.recall_score(is_relevant_results, recall_predictions)
        recall_scores.append(f"{recall_score}@{k}")
        print(f"Recall Score: {recall_score}@{k}")

        # TODO: should we only append content when it is relevant?
        content.append(doc['content']) 

    eval_metrics = {
        "recall_scores": recall_scores,
        "precision_scores": precision_scores,
    }

    return content, eval_metrics