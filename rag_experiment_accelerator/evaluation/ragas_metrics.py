from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from rag_experiment_accelerator.llm.prompt import (
    ragas_answer_relevance_instruction,
    ragas_context_recall_instruction,
    ragas_context_precision_instruction,
)
from rag_experiment_accelerator.llm.response_generator import ResponseGenerator
from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


class RagasEvals:
    """Class that leverages the evaluators from the ragas evaluation framework
    for RAG pipelines: https://github.com/explodinggradients/ragas
    """
    def __init__(self, response_generator: ResponseGenerator):
        self.response_generator = response_generator

    def compute_score(
            self,
            metric_type: str,
            question: str,
            generated_answer: str,
            ground_truth_answer: str,
            retrieved_contexts: list[str]) -> float:
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
        match metric_type:
            case "ragas_answer_relevance":
                score = self.ragas_answer_relevance(question=question, answer=generated_answer)
            case "ragas_context_precision":
                score = self.ragas_context_precision(question=question, retrieved_contexts=retrieved_contexts)
            case "ragas_context_recall":
                score = self.ragas_context_recall(question=question,
                                                  groundtruth_answer=ground_truth_answer,
                                                  retrieved_contexts=retrieved_contexts)
            case _:
                raise KeyError(f"Invalid metric type: {metric_type}")

        return score

    def lower_and_strip(self, text: str) -> str:
        return text.lower().strip()

    def ragas_answer_relevance(self, question, answer) -> float:
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
        result = self.response_generator.generate_response(
            ragas_answer_relevance_instruction, text=answer
        )
        if result is None:
            logger.warning("Unable to generate answer relevance score")
            return 0.0

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        embedding1 = model.encode([str(question)])
        embedding2 = model.encode([str(result)])
        similarity_score = cosine_similarity(embedding1, embedding2)

        return float(similarity_score[0][0] * 100)

    def ragas_context_precision(
        self, question: str, retrieved_contexts: list[str]
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
            result: str | None = self.response_generator.generate_response(
                ragas_context_precision_instruction,
                context=context,
                question=question,
            )
            llm_judge_response = self.lower_and_strip(result)
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

    def ragas_context_recall(
        self, question: str, groundtruth_answer: str, retrieved_contexts: list[str]
    ) -> float:
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
        prompt = (
            "\nquestion: "
            + question
            + "\ncontext: "
            + context
            + "\nanswer: "
            + groundtruth_answer
        )
        result = self.response_generator.generate_response(
            sys_message=ragas_context_recall_instruction,
            prompt=prompt,
        )
        good_response = '"Attributed": "1"'
        bad_response = '"Attributed": "0"'

        return (
            result.count(good_response)
            / (result.count(good_response) + result.count(bad_response))
        ) * 100