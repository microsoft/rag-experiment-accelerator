from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from rag_experiment_accelerator.llm.prompt import (
    llm_answer_relevance_instruction,
    llm_context_recall_instruction,
    llm_context_precision_instruction,
)
from rag_experiment_accelerator.llm.response_generator import ResponseGenerator
from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


def lower_and_strip(text):
    """
    Converts the input to lowercase without spaces or empty string if None.

    Args:
        text (str): The string to format.

    Returns:
        str: The formatted input string.
    """
    if text is None:
        return ""
    else:
        return text.lower().strip()


def llm_answer_relevance(
    response_generator: ResponseGenerator, question, answer
) -> float:
    """
    Scores the relevancy of the answer according to the given question.
    Answers with incomplete, redundant or unnecessary information is penalized.
    Score can range from 0 to 1 with 1 being the best.

    Args:
        question (str): The question being asked.
        answer (str): The generated answer.

    Returns:
        double: The relevancy score generated between the question and answer.

    """
    result = response_generator.generate_response(
        llm_answer_relevance_instruction, text=answer
    )
    if result is None:
        logger.warning("Unable to generate answer relevance score")
        return 0.0

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    embedding1 = model.encode([str(question)])
    embedding2 = model.encode([str(result)])
    similarity_score = cosine_similarity(embedding1, embedding2)

    return float(similarity_score[0][0] * 100)


def llm_context_precision(
    response_generator: ResponseGenerator, question, retrieved_contexts
) -> float:
    """
    Computes precision by assessing whether each retrieved context is useful for answering a question.
    Only considers the presence of relevant chunks in the retrieved contexts, but doesn't take into
    account their ranking order.

    Args:
        question (str): The question being asked.
        retrieved_contexts (list[str]): The list of retrieved contexts for the query.

    Returns:
        double: proportion of relevant chunks retrieved for the question
    """
    relevancy_scores = []

    for context in retrieved_contexts:
        result: str | None = response_generator.generate_response(
            llm_context_precision_instruction,
            context=context,
            question=question,
        )
        llm_judge_response = lower_and_strip(result)
        # Since we're only asking for one response, the result is always a boolean 1 or 0
        if llm_judge_response == "yes":
            relevancy_scores.append(1)
        elif llm_judge_response == "no":
            relevancy_scores.append(0)
        else:
            logger.warning("Unable to generate context precision score")

    logger.debug(relevancy_scores)

    if not relevancy_scores:
        logger.warning("Unable to compute average context precision")
        return -1
    else:
        return (sum(relevancy_scores) / len(relevancy_scores)) * 100


def llm_context_recall(
    response_generator: ResponseGenerator,
    question,
    groundtruth_answer,
    retrieved_contexts,
):
    """
    Estimates context recall by estimating TP and FN using annotated answer (ground truth) and retrieved context.
    Context_recall values range between 0 and 1, with higher values indicating better performance.
    To estimate context recall from the ground truth answer, each sentence in the ground truth answer is analyzed to determine
    whether it can be attributed to the retrieved context or not. In an ideal scenario, all sentences in the ground truth answer
    should be attributable to the retrieved context. The formula for calculating context recall is as follows:
    context_recall = GT sentences that can be attributed to context / nr sentences in GT

    Code adapted from https://github.com/explodinggradients/ragas
    Copyright [2023] [Exploding Gradients]
    under the Apache License (see evaluation folder)

    Args:
        question (str): The question being asked
        groundtruth_answer (str): The ground truth ("output_prompt")
        retrieved_contexts (list[str]): The list of retrieved contexts for the query

    Returns:
        double: The context recall score generated between the ground truth (expected) and context.
    """
    context = "\n".join(retrieved_contexts)

    result: list | None = response_generator.generate_response(
        llm_context_recall_instruction,
        context=context,
        question=question,
        answer=groundtruth_answer,
    )

    good_responses = 0

    for response in result:
        try:
            score = response.get("attributed", 0)
            good_responses += int(score)
        except ValueError:
            logger.warning(f"Unable to parse {score} as int.")
    if not result:
        return -1
    else:
        return (good_responses / len(result)) * 100


def compute_llm_based_score(
    metric_type,
    question,
    actual,
    expected,
    response_generator: ResponseGenerator,
    retrieved_contexts,
):
    match metric_type:
        case "llm_answer_relevance":
            score = llm_answer_relevance(response_generator, question, actual)
        case "llm_context_precision":
            score = llm_context_precision(
                response_generator, question, retrieved_contexts
            )
        case "llm_context_recall":
            score = llm_context_recall(
                response_generator, question, expected, retrieved_contexts
            )
        case _:
            raise KeyError(f"Invalid metric type: {metric_type}")

    return score
