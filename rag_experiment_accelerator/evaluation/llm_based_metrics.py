from rag_experiment_accelerator.evaluation.ragas_metrics import RagasEvals
from rag_experiment_accelerator.evaluation.azure_ai_quality_metrics import PromptFlowEvals
from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


def compute_llm_based_score(
    metric_type: str,
    question: str,
    generated_answer: str,
    ground_truth_answer: str,
    retrieved_contexts: list[str],
    ragas_evals: RagasEvals,
    pf_evals: PromptFlowEvals,
) -> float:
    """
    Compute the LLM-as-a-judge score for the given metric type.

    Args:
        metric_type (str): The metric type to compute the score for.
        question (str): The question.
        generated_answer (str): The generated answer.
        ground_truth_answer (str): The ground truth answer.
        retrieved_contexts (List[str]): The retrieved contexts.
        response_generator (ResponseGenerator): The response generator.
        ragas_evals (RagasEvals): The ragas evaluators.
        pf_evals (PromptFlowEvals): The promptflow evaluators.

    Returns:
        float: The computed LLM-as-a-judge score.
    """
    if metric_type.startswith("ragas_"):
        score = ragas_evals.compute_score(
            metric_type=metric_type,
            question=question,
            generated_answer=generated_answer,
            ground_truth_answer=ground_truth_answer,
            retrieved_contexts=retrieved_contexts,
        )
    elif metric_type.startswith("pf_"):
        score = pf_evals.compute_score(
            metric_name=metric_type,
            question=question,
            generated_answer=generated_answer,
            ground_truth_answer=ground_truth_answer,
            retrieved_contexts=retrieved_contexts,
        )
    else:
        raise KeyError(f"Invalid metric type: {metric_type}")

    return score
