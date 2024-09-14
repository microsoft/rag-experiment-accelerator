from rag_experiment_accelerator.config.environment import Environment
from promptflow.core import AzureOpenAIModelConfiguration
from promptflow_evals import (
    CoherenceEvaluator,
    FluencyEvaluator,
    GroundednessEvaluator,
    RelevanceEvaluator,
    SimilarityEvaluator
)


class PromptFlowEvals:
    def __init__(self, environment: Environment, deployment_name: str):
        self.openai_deployment_name = deployment_name
        self.openai_endpoint = environment.openai_endpoint
        self.openai_api_key = environment.openai_api_key

        self.model_config = AzureOpenAIModelConfiguration(
                azure_endpoint=self.openai_endpoint,
                api_key=self.openai_api_key,
                azure_deployment=self.openai_deployment_name
        )

    def relevance_evaluator(self):
        """
        Relevance measures how well the answer addresses the main aspects of the question, based on the context.
        Consider whether all and only the important aspects are contained in the answer when evaluating relevance.
        Given the context and question, score the relevance of the answer between one to five stars using the
        following rating scale:
            One star: the answer completely lacks relevance
            Two stars: the answer mostly lacks relevance
            Three stars: the answer is partially relevant
            Four stars: the answer is mostly relevant
            Five stars: the answer has perfect relevance
        """
        return RelevanceEvaluator(
            model_config=self.model_config
        )

    def coherence_evaluator(self):
        return CoherenceEvaluator(
            model_config=self.model_config
        )

    def similarity_evaluator(self):
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
        return SimilarityEvaluator(
            model_config=self.model_config
        )

    def groundedness_evaluator(self):
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
        """
        return GroundednessEvaluator(
            model_config=self.model_config
        )

    def fluency_evaluator(self):
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
        return FluencyEvaluator(
            model_config=self.model_config
        )
