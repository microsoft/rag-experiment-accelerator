from rag_experiment_accelerator.llm.response_generator import ResponseGenerator
from rag_experiment_accelerator.evaluation.ragas_evaluators import RagasEvals
from rag_experiment_accelerator.evaluation.promptflow_evaluators import PromptFlowEvals
from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


def compute_ragas_scores(
    metric_type: str,
    question: str,
    generated_answer: str,
    ground_truth_answer: str,
    retrieved_contexts: list[str],
    response_generator: ResponseGenerator
):
    """
    Compute the LLM-as-a-judge score for the given metric type from the RAGAS framework.

    Args:
        metric_type (str): The metric type to compute the score for.
        question (str): The question.
        generated_answer (str): The generated answer.
        ground_truth_answer (str): The ground truth answer.
        retrieved_contexts (List[str]): The retrieved contexts.
        response_generator (ResponseGenerator): The response generator.

    Returns:
        float: The computed LLM-as-a-judge score.
    """
    r_eval = RagasEvals(response_generator)
    match metric_type:
        case "ragas_answer_relevance":
            score = r_eval.ragas_answer_relevance(question=question, answer=generated_answer)
        case "ragas_context_precision":
            score = r_eval.ragas_context_precision(question=question, retrieved_contexts=retrieved_contexts)
        case "ragas_context_recall":
            score = r_eval.ragas_context_recall(question=question, 
                                                groundtruth_answer=ground_truth_answer, 
                                                retrieved_contexts=retrieved_contexts)
        case _:
            raise KeyError(f"Invalid metric type: {metric_type}")

    return score


def compute_promptflow_evals_scores(
    metric_type: str,
    question: str,
    generated_answer: str,
    ground_truth_answer: str,
    retrieved_contexts: list[str],
    pf_evals: PromptFlowEvals
):
    """
    Compute the LLM-as-a-judge score for the given metric type from the PromptFlowEvals framework.

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
    pf_eval = RagasEvaluator(response_generator)
    match metric_type == "pf_answer_relevance":
        # score relevance for each context separately and take the max
        scores = []
        for search_doc in retrieved_results:
            result = pf_evals.relevance_evaluator()(
                question=question, answer=generated_answer, context=search_doc
            )
            score = result["gpt_relevance"]
            scores.append(score)
        score = max(scores)
    elif metric_type == "gen_gpt_answer_coherence":
        result = pf_evals.coherence_evaluator()(
            question=question, answer=generated_answer
        )
        score = result["gpt_coherence"]
    elif metric_type == "gen_gpt_answer_fluency":
        result = pf_evals.fluency_evaluator()(
            question=question, answer=generated_answer
        )
        score = result["gpt_fluency"]
    elif metric_type == "gen_gpt_answer_groundedness":
        # use all retrieved contexts for groundedness evaluation
        scores = []
        for search_doc in retrieved_results:
            result = pf_evals.groundedness_evaluator()(
                answer=generated_answer, context=search_doc
            )
            score = result["gpt_groundedness"]
            scores.append(score)
        score = max(scores)
    elif metric_type == "gen_gpt_gt_answer_similarity":
        result = pf_evals.similarity_evaluator()(
            question=question, answer=generated_answer, ground_truth=ground_truth_answer
        )
        score = result["gpt_similarity"]
)
