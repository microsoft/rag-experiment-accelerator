from rag_experiment_accelerator.llm.response_generator import ResponseGenerator
from rag_experiment_accelerator.evaluation.ragas_evaluator import RagasEvaluator
from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


def compute_llm_based_score(
    metric_type,
    question,
    actual,
    expected,
    retrieved_contexts,
    response_generator: ResponseGenerator
):
    """
    Compute the LLM-as-a-judge score for the given metric type from the RAGAS framework.

    Args:
        metric_type (str): The metric type to compute the score for.
        question (str): The question.
        actual (str): The actual answer.
        expected (str): The expected answer.
        retrieved_contexts (List[str]): The retrieved contexts.
        response_generator (ResponseGenerator): The response generator.

    Returns:
        float: The computed LLM-as-a-judge score.
    """
    r_eval = RagasEvaluator(response_generator)
    match metric_type:
        case "ragas_answer_relevance":
            score = r_eval.ragas_answer_relevance(question, actual)
        case "ragas_context_precision":
            score = r_eval.ragas_context_precision(question, retrieved_contexts)
        case "ragas_context_recall":
            score = r_eval.ragas_context_recall(question, expected, retrieved_contexts)
        case _:
            raise KeyError(f"Invalid metric type: {metric_type}")

    return score
