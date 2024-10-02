from azure.ai.evaluation import (
    CoherenceEvaluator,
    FluencyEvaluator,
    GroundednessEvaluator,
    RelevanceEvaluator,
    SimilarityEvaluator,
)
from promptflow.core import AzureOpenAIModelConfiguration


class AzureAIEvals:
    """Class that leverages the evaluators from the Promptflow evaluation framework
    for LLM pipelines"""
    def __init__(self, az_openai_model_config: AzureOpenAIModelConfiguration):
        self.model_config = az_openai_model_config

    def compute_score(
        self,
        metric_name: str,
        question: str,
        generated_answer: str,
        ground_truth_answer: str,
        retrieved_contexts: list[str],
    ) -> float:
        """
        Compute LLM as a judge score based on the Promptflow evaluation framework.
        """
        match metric_name:
            case "pf_answer_relevance":
                score = self.relevance_evaluator(
                    question=question, answer=generated_answer
                )
            case "pf_answer_coherence":
                score = self.coherence_evaluator(
                    question=question, answer=generated_answer
                )
            case "pf_answer_similarity":
                score = self.similarity_evaluator(
                    question=question,
                    answer=generated_answer,
                    ground_truth=ground_truth_answer,
                )
            case "pf_answer_fluency":
                score = self.fluency_evaluator(
                    question=question, answer=generated_answer
                )
            case "pf_answer_groundedness":
                score = self.groundedness_evaluator(
                    answer=generated_answer, retrieved_contexts=retrieved_contexts
                )
            case _:
                raise KeyError(f"Invalid metric type: {metric_name}")

        return score

    def relevance_evaluator(self, question: str, answer: str) -> float:
        eval_fn = RelevanceEvaluator(model_config=self.model_config)
        score = eval_fn(question=question, answer=answer)
        return score

    def coherence_evaluator(self, question: str, answer: str) -> float:
        eval_fn = CoherenceEvaluator(model_config=self.model_config)
        score = eval_fn(question=question, answer=answer)
        return score

    def similarity_evaluator(
        self, question: str, answer: str, ground_truth: str
    ) -> float:
        """
        Equivalence, as a metric, measures the similarity between the predicted answer and the correct answer.
        If the information and content in the predicted answer is similar or equivalent to the correct answer,
        then the value of the Equivalence metric should be high, else it should be low. Given the question,
        correct answer, and predicted answer, determine the value of Equivalence metric using the following
        rating scale:
            One star: the predicted answer is not at all similar to the correct answer
            Two stars: the predicted answer is mostly not similar to the correct answer
            Three stars: the predicted answer is somewhat similar to the correct answer
            Four stars: the predicted answer is mostly similar to the correct answer
            Five stars: the predicted answer is completely similar to the correct answer

        This rating value should always be an integer between 1 and 5.
        """
        eval_fn = SimilarityEvaluator(model_config=self.model_config)
        score = eval_fn(question=question, answer=answer, ground_truth=ground_truth)
        return score

    def fluency_evaluator(self, question: str, answer: str) -> float:
        """
        Fluency measures the quality of individual sentences in the answer,
        and whether they are well-written and grammatically correct. Consider
        the quality of individual sentences when evaluating fluency. Given the
        question and answer, score the fluency of the answer between one to
        five stars using the following rating scale:
            One star: the answer completely lacks fluency
            Two stars: the answer mostly lacks fluency
            Three stars: the answer is partially fluent
            Four stars: the answer is mostly fluent
            Five stars: the answer has perfect fluency

        This rating value should always be an integer between 1 and 5.
        """
        eval_fn = FluencyEvaluator(model_config=self.model_config)
        score = eval_fn(question=question, answer=answer)
        return score

    def groundedness_evaluator(
        self, answer: str, retrieved_contexts: list[str]
    ) -> float:
        """
        Groundedness is measured the following way:
        Given a CONTEXT and an ANSWER about that CONTEXT, rate the following way if the ANSWER is
        entailed by the CONTEXT,
            1. 5: The ANSWER follows logically from the information contained in the CONTEXT.
            2. 1: The ANSWER is logically false from the information contained in the CONTEXT.
            3. an integer score between 1 and 5 and if such integer score does not exist, use 1:
                It is not possible to determine whether the ANSWER is true or false without
                further information. Read the passage of information thoroughly and select the
                correct answer from the three answer labels. Read the CONTEXT thoroughly to
                ensure you know what the CONTEXT entails.

        This rating value should always be an integer between 1 and 5.

        Here we have a list of contexts and an answer. We return the best (max) groundedness score
        when comparing the answer with each context in the list.

        Args:
            answer (str): The answer generated by the model.
            retrieved_contexts (list[str]): The list of retrieved contexts for the query.

        Returns:
            float: The groundedness score generated between the answer and the list of contexts
        """
        eval_fn = GroundednessEvaluator(model_config=self.model_config)

        best_score = 0
        for context in retrieved_contexts:
            score = eval_fn(context=context, answer=answer)
            best_score = max(best_score, score)

        return best_score
