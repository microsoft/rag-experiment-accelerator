from evaluation.spacy_evaluator import SpacyEvaluator


def get_recall_score(is_relevant_results: list[bool], total_relevant_docs: int):
    if total_relevant_docs == 0: 
        return 0

    num_of_relevant_docs = is_relevant_results.count(True)
    
    return num_of_relevant_docs/total_relevant_docs

def get_precision_score(is_relevant_results: list[bool]):
    num_of_recommended_docs = len(is_relevant_results)
    if num_of_recommended_docs == 0: 
        return 0
    num_of_relevant_docs = is_relevant_results.count(True)

    return num_of_relevant_docs/num_of_recommended_docs


def evaluate_search_result(search_response: list, evaluation_content: str, evaluator: SpacyEvaluator):
    content = []

    # create list of all docs with their is_relevant result to calculate recall
    total_relevent_docs = []
    for doc in search_response:
        is_relevant = evaluator.is_relevant(doc["content"], evaluation_content)
        total_relevent_docs.append(is_relevant)

    recall_scores = []
    precision_scores = []
    is_relevant_results: list[bool] = []
    for i, doc in enumerate(search_response):
        k = i + 1
        print("++++++++++++++++++++++++++++++++++")
        print(f"Content: {doc['content']}")
        print(f"Search Score: {doc['@search.score']}")

        is_relevant = total_relevent_docs[i]
        is_relevant_results.append(is_relevant)

        precision_score = get_precision_score(is_relevant_results)
        precision_scores.append(f"{precision_score}@{k}")
        print(f"Precision Score: {precision_score}@{k}")

        recall_score = get_recall_score(is_relevant_results, sum(total_relevent_docs))
        recall_scores.append(f"{recall_score}@{k}")
        print(f"Recall Score: {recall_score}@{k}")

        # TODO: should we only append content when it is relevant?
        content.append(doc['content']) 

    metrics = {
        "recall_scores": recall_scores,
        "precision_scores": precision_scores,
    }

    return content, metrics