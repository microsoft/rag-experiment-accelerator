from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import os
import warnings

import mlflow
import pandas as pd

from dotenv import load_dotenv
from numpy import mean

from rag_experiment_accelerator.artifact.handlers.query_output_handler import (
    QueryOutputHandler,
)
from rag_experiment_accelerator.config.config import Config
from rag_experiment_accelerator.config.index_config import IndexConfig
from promptflow.core import AzureOpenAIModelConfiguration
from rag_experiment_accelerator.evaluation import plain_metrics
from rag_experiment_accelerator.evaluation.llm_based_metrics import (
    compute_llm_based_score,
)
from rag_experiment_accelerator.evaluation.plot_metrics import (
    draw_hist_df,
    draw_search_chart,
    generate_metrics,
    plot_apk_scores,
    plot_map_scores,
    plot_mapk_scores,
)
from rag_experiment_accelerator.evaluation.transformer_based_metrics import (
    compute_transformer_based_score,
)

from rag_experiment_accelerator.llm.response_generator import ResponseGenerator
from rag_experiment_accelerator.evaluation.ragas_metrics import RagasEvals
from rag_experiment_accelerator.evaluation.promptflow_quality_metrics import PromptFlowEvals
from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.config.environment import Environment

logger = get_logger(__name__)

load_dotenv()
warnings.filterwarnings("ignore")


def lower_and_strip(text: str) -> str:
    return text.lower().strip()


def compute_metrics(
    metric_type,
    question,
    actual,
    expected,
    retrieved_contexts,
    ragas_evals: RagasEvals,
    pf_evals: PromptFlowEvals
):
    """
    Computes a score for the similarity between two strings using a specified metric.

    Args:
        metric_type (str): The type of metric to use for comparison. Valid options are:
            - "lcsstr": Longest common substring
            - "lcsseq": Longest common subsequence
            - "jaro_winkler": Jaro-Winkler distance
            - "hamming": Hamming distance
            - "jaccard": Jaccard similarity
            - "levenshtein": Levenshtein distance
            - "fuzzy_score": RapidFuzz similarity. This is faster than the associated function in FuzzyWuzzy.
                             Default match type is "token_set_ratio".
            - "cosine_ochiai": Cosine similarity (Ochiai coefficient)
            - "rouge1_precision": The ROUGE-1 precision score. This is the number of overlapping unigrams
                                  between the actual and expected strings divided by the number of unigrams
                                  in the expected string.
            - "rouge1_recall": The ROUGE-1 recall score. This is the number of overlapping unigrams between
                               the actual and expected strings divided by the number of unigrams in the actual string.
            - "rouge1_fmeasure": ROUGE-1 F1 score. This is the harmonic mean of the ROUGE-1 precision and recall scores.
            - "rouge2_precision": The ROUGE-2 precision score. This is the number of overlapping bigrams between
                                    the actual and expected strings divided by the number of bigrams in the expected string.
            - "rouge2_recall": The ROUGE-2 recall score. This is the number of overlapping bigrams between the actual
                               and expected strings divided by the number of bigrams in the actual string.
            - "rouge2_fmeasure": ROUGE-2 F1 score. This is the harmonic mean of the ROUGE-2 precision and recall scores.
            - "rougeL_precision": The ROUGE-L precision score is the length of overlapping longest common subsequence
                                  between the actual and expected strings divided by the number of unigrams
                                  in the predicted string.
            - "rougeL_recall": The ROUGE-L recall score is the length of overlapping longest common subsequence
                               between the actual and expected strings divided by the number of unigrams in the
                               actual string.
            - "rougeL_fmeasure": ROUGE-L F1 score. This is the harmonic mean of the ROUGE-L precision and recall scores.
            - "bert_all_MiniLM_L6_v2": BERT-based semantic similarity (MiniLM L6 v2 model)
            - "bert_base_nli_mean_tokens": BERT-based semantic similarity (base model, mean tokens)
            - "bert_large_nli_mean_tokens": BERT-based semantic similarity (large model, mean tokens)
            - "bert_large_nli_stsb_mean_tokens": BERT-based semantic similarity (large model, STS-B, mean tokens)
            - "bert_distilbert_base_nli_stsb_mean_tokens": BERT-based semantic similarity (DistilBERT base model, STS-B, mean tokens)
            - "bert_paraphrase_multilingual_MiniLM_L12_v2": BERT-based semantic similarity (multilingual paraphrase model, MiniLM L12 v2)
            - "ragas_context_precision": Verifies whether or not a given context is useful for answering a question.
            - "ragas_answer_relevance": Scores the relevancy of the answer according to the given question.
            - "ragas_context_recall": Scores context recall by estimating TP and FN using annotated answer (ground truth) and retrieved context.
            - "pf_answer_relevance": Scores the relevancy of the answer according to the given question.
            - "pf_answer_coherence": Scores the coherence of the answer according to the given question.
            - "pf_answer_similarity": Scores the similarity of the answer to the ground truth answer.
            - "pf_answer_fluency": Scores the fluency of the answer according to the given question.
            - "pf_answer_groundedness": Scores the groundedness of the answer according to the retrieved contexts.
        question (str): question text
        actual (str): The first string to compare.
        expected (str): The second string to compare.
        retrieved_contexts (list[str]): The list of retrieved contexts for the query.
        ragas_evals (RagasEvals): The Ragas evaluators to use for scoring.
        pf_evals (PromptFlowEvals): The PromptFlow evaluators to use for scoring.


    Returns:
        float: The similarity score between the two strings, as determined by the specified metric.
    """

    if metric_type.startswith("rouge"):
        return plain_metrics.rouge_score(ground_truth=expected, prediction=actual, rouge_metric_name=metric_type)
    else:
        plain_metric_func = getattr(plain_metrics, metric_type, None)
        if plain_metric_func:
            return plain_metric_func(actual, expected)

    try:
        score = compute_transformer_based_score(actual, expected, metric_type)
    except KeyError:
        try:
            score = compute_llm_based_score(
                metric_type,
                question,
                actual,
                expected,
                retrieved_contexts,
                ragas_evals,
                pf_evals
            )
        except KeyError:
            logger.error(f"Unsupported metric type: {metric_type}")

    return score


def evaluate_single_prompt(
    data,
    ragas_evals,
    pf_evals,
    metric_types,
    data_list,
    total_precision_scores_by_search_type,
    map_scores_by_search_type,
    average_precision_for_search_type,
):
    actual = lower_and_strip(data.actual)
    expected = lower_and_strip(data.expected)

    metric_dic = {}

    for metric_type in metric_types:
        score = compute_metrics(
            metric_type,
            data.question,
            actual,
            expected,
            data.retrieved_contexts,
            ragas_evals,
            pf_evals
        )
        metric_dic[metric_type] = score

    metric_dic["question"] = data.question
    metric_dic["retrieved_contexts"] = data.retrieved_contexts
    metric_dic["actual"] = actual
    metric_dic["expected"] = expected
    metric_dic["search_type"] = data.search_type
    data_list.append(metric_dic)

    if not total_precision_scores_by_search_type.get(data.search_type):
        total_precision_scores_by_search_type[data.search_type] = {}
        map_scores_by_search_type[data.search_type] = []
        average_precision_for_search_type[data.search_type] = []
    for eval in data.search_evals:
        scores = eval.get("precision_scores")
        if scores:
            average_precision_for_search_type[data.search_type].append(mean(scores))
        for i, score in enumerate(scores):
            if total_precision_scores_by_search_type[data.search_type].get(i + 1):
                total_precision_scores_by_search_type[data.search_type][i + 1].append(
                    score
                )
            else:
                total_precision_scores_by_search_type[data.search_type][i + 1] = [score]


def evaluate_prompts(
    environment: Environment,
    config: Config,
    index_config: IndexConfig,
    mlflow_client: mlflow.MlflowClient,
    name_suffix: str,
):
    """
    Evaluates prompts using various metrics and logs the results to MLflow.

    Args:
        environment (Environment): Initialized Environment class containing environment configuration
        config (Config): The configuration settings to use for evaluation.
        index_config (IndexConfig): Parameters of the index such as chunking and embedding model.
        mlflow_client (mlflow.MlflowClient): The MLflow client to use for logging the results.
        name_suffix (str): Name suffix to use for all outputs created.

    Returns:
        None
    """
    metric_types = config.eval.metric_types
    num_search_type = config.search.search_type
    data_list = []

    pd.set_option("display.max_columns", None)

    total_precision_scores_by_search_type = {}
    map_scores_by_search_type = {}
    average_precision_for_search_type = {}

    handler = QueryOutputHandler(config.path.query_data_dir)

    # Ragas and PromptFlow evaluators
    response_generator = ResponseGenerator(
        environment, config, config.openai.azure_oai_eval_deployment_name
    )
    ragas_evals = RagasEvals(response_generator)

    az_openai_model_config = AzureOpenAIModelConfiguration(
        azure_endpoint=environment.openai_endpoint,
        api_key=environment.openai_api_key,
        azure_deployment=config.openai.azure_oai_eval_deployment_name
    )

    pf_evals = PromptFlowEvals(az_openai_model_config)

    query_data_load = handler.load(
        index_config.index_name(), config.experiment_name, config.job_name
    )
    question_count = query_data_load[0].question_count

    with ExitStack() as stack:
        executor = stack.enter_context(ThreadPoolExecutor(config.max_worker_threads))
        futures = {
            executor.submit(
                evaluate_single_prompt,
                data,
                ragas_evals,
                pf_evals,
                metric_types,
                data_list,
                total_precision_scores_by_search_type,
                map_scores_by_search_type,
                average_precision_for_search_type,
            ): data
            for data in query_data_load
        }

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                logger.error(f"Evaluate query line generated an exception: {exc}")

    eval_scores_df = {"search_type": [], "k": [], "score": [], "map_at_k": []}

    for (
        search_type,
        scores_at_k,
    ) in total_precision_scores_by_search_type.items():
        for k, scores in scores_at_k.items():
            avg_at_k = mean(scores)
            # not sure if this would be problematic or not.
            eval_scores_df["search_type"].append(search_type)
            eval_scores_df["k"].append(k)
            eval_scores_df["score"].append(avg_at_k)
            mean_at_k = mean(eval_scores_df["score"][:k])
            eval_scores_df["map_at_k"].append(mean_at_k)

    mean_scores = {"search_type": [], "mean": []}

    for search_type, scores in average_precision_for_search_type.items():
        mean_scores["search_type"].append(search_type)
        mean_scores["mean"].append(mean(scores))

    run_id = mlflow.active_run().info.run_id
    columns_to_remove = ["question", "retrieved_contexts", "actual", "expected"]
    additional_columns_to_remove = ["search_type"]
    df = pd.DataFrame(data_list)
    df.to_csv(
        os.path.join(config.path.eval_data_dir, f"{name_suffix}.csv"), index=False
    )
    logger.debug(f"Eval scores: {df.head()}")

    temp_df = df.drop(columns=columns_to_remove)
    draw_search_chart(temp_df, run_id, mlflow_client)

    temp_df = temp_df.drop(columns=additional_columns_to_remove)

    if isinstance(num_search_type, str):
        num_search_type = [num_search_type]
    sum_all_columns = temp_df.sum() / (question_count * len(num_search_type))
    sum_df = pd.DataFrame([sum_all_columns], columns=temp_df.columns)

    sum_dict = {}
    for col_name in sum_df.columns:
        sum_dict[col_name] = float(sum_df[col_name].values)

    sum_df.to_csv(
        os.path.join(config.path.eval_data_dir, f"sum_{name_suffix}.csv"), index=False
    )

    ap_scores_df = pd.DataFrame(eval_scores_df)
    ap_scores_df.to_csv(
        os.path.join(
            config.path.eval_data_dir, f"{name_suffix}_ap_scores_at_k_test.csv"
        ),
        index=False,
    )
    plot_apk_scores(ap_scores_df, run_id, mlflow_client)
    plot_mapk_scores(ap_scores_df, run_id, mlflow_client)

    map_scores_df = pd.DataFrame(mean_scores)
    map_scores_df.to_csv(
        os.path.join(config.path.eval_data_dir, f"{name_suffix}_map_scores_test.csv"),
        index=False,
    )
    plot_map_scores(map_scores_df, run_id, mlflow_client)

    common_data = query_data_load[0]
    mlflow.log_param("question_count", common_data.question_count)
    mlflow.log_param("retrieve_num_of_documents", common_data.retrieve_num_of_documents)
    mlflow.log_param("cross_encoder_at_k", common_data.cross_encoder_at_k)
    mlflow.log_param("chunk_overlap", index_config.chunking.overlap_size)
    mlflow.log_param(
        "embedding_dimension",
        config.get_embedding_model(index_config.embedding_model.model_name).dimension,
    )
    mlflow.log_param("embedding_model_name", index_config.embedding_model.model_name)
    mlflow.log_param("ef_construction", index_config.ef_construction)
    mlflow.log_param("ef_search", index_config.ef_search)
    mlflow.log_param("run_metrics", sum_dict)
    mlflow.log_metrics(sum_dict)
    mlflow.log_artifact(os.path.join(config.path.eval_data_dir, f"{name_suffix}.csv"))
    mlflow.log_artifact(
        os.path.join(config.path.eval_data_dir, f"sum_{name_suffix}.csv")
    )
    draw_hist_df(sum_df, run_id, mlflow_client)
    generate_metrics(config.experiment_name, run_id, mlflow_client)
    mlflow.end_run()
