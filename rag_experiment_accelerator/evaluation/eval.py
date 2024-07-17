import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import os
import warnings

import evaluate
import mlflow
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import textdistance
from dotenv import load_dotenv
from fuzzywuzzy import fuzz
from numpy import mean
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from rag_experiment_accelerator.artifact.handlers.query_output_handler import (
    QueryOutputHandler,
)
from rag_experiment_accelerator.config.config import Config
from rag_experiment_accelerator.config.index_config import IndexConfig
from rag_experiment_accelerator.llm.prompts import (
    llm_answer_relevance_instruction,
    llm_context_recall_instruction,
    llm_context_precision_instruction,
)
from rag_experiment_accelerator.llm.response_generator import ResponseGenerator
from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.config.environment import Environment

logger = get_logger(__name__)

load_dotenv()
warnings.filterwarnings("ignore")

algs = textdistance.algorithms


def lower(text):
    """
    Converts the input text to lowercase.

    Args:
        text (str): The text to convert to lowercase.

    Returns:
        str: The input text in lowercase.
    """
    return text.lower()


def remove_spaces(text):
    """
    Removes leading and trailing spaces from a string.

    Args:
        text (str): The string to remove spaces from.

    Returns:
        str: The input string with leading and trailing spaces removed.
    """
    return text.strip()


# https://huggingface.co/spaces/evaluate-metric/bleu
def bleu(predictions, references):
    bleu = evaluate.load("bleu")

    results = bleu.compute(predictions=predictions, references=references, max_order=2)
    # multiplying by 100 to maintain consistency with previous implementation
    return results["bleu"] * 100


def fuzzy(doc1, doc2):
    """
    Calculates the fuzzy score between two documents.

    Parameters:
    doc1 (str): The first document to compare.
    doc2 (str): The second document to compare.

    Returns:
    int: The fuzzy score between the two documents.
    """
    differences = []
    fuzzy_compare_values(doc1, doc2, differences)

    return int(sum(differences) / len(differences))


def fuzzy_compare_values(value1, value2, differences):
    """
    Compares two values using fuzzy string matching and appends the similarity score to a list of differences.

    Args:
        value1 (str): The first value to compare.
        value2 (str): The second value to compare.
        differences (list): A list to which the similarity score will be appended.

    Returns:
        None
    """
    similarity_score = fuzz.token_set_ratio(value1, value2)
    differences.append(similarity_score)


def compare_semantic_document_values(doc1, doc2, model_type):
    """
    Compares the semantic values of two documents and returns the percentage of differences.

    Args:
        doc1 (str): The first document to compare.
        doc2 (str): The second document to compare.
        model_type (SentenceTransformer): The SentenceTransformer model to use for comparison.

    Returns:
        int: The percentage of differences between the two documents.
    """
    differences = []
    semantic_compare_values(doc1, doc2, differences, model_type)

    return int(sum(differences) / len(differences))


def semantic_compare_values(
    value1: str,
    value2: str,
    differences: list[float],
    model_type: SentenceTransformer,
) -> None:
    """
    Computes the semantic similarity between two values using a pre-trained SentenceTransformer model.

    Args:
        value1 (str): The first value to compare.
        value2 (str): The second value to compare.
        differences (list[float]): A list to store the similarity scores.
        model_type (SentenceTransformer): The pre-trained SentenceTransformer model to use for encoding the values.

    Returns:
        None
    """
    embedding1 = model_type.encode([str(value1)])
    embedding2 = model_type.encode([str(value2)])
    similarity_score = cosine_similarity(embedding1, embedding2)

    differences.append(similarity_score * 100)


def levenshtein(value1, value2):
    """
    Calculates the Levenshtein distance between two strings and returns the normalized similarity score as a percentage.

    Args:
        value1 (str): The first string to compare.
        value2 (str): The second string to compare.

    Returns:
        int: The normalized similarity score as a percentage.
    """
    score = int(algs.levenshtein.normalized_similarity(value1, value2) * 100)
    return score


def jaccard(value1, value2):
    """
    Calculates the Jaccard similarity score between two sets of values.

    Args:
        value1 (set): The first set of values.
        value2 (set): The second set of values.

    Returns:
        int: The Jaccard similarity score between the two sets of values, as a percentage.
    """
    score = int(algs.jaccard.normalized_similarity(value1, value2) * 100)
    return score


def hamming(value1, value2):
    """
    Calculates the Hamming similarity score between two values.

    Args:
        value1 (str): The first value to compare.
        value2 (str): The second value to compare.

    Returns:
        int: The Hamming similarity score between the two values, as a percentage.
    """
    score = int(algs.hamming.normalized_similarity(value1, value2) * 100)
    return score


def jaro_winkler(value1, value2):
    """
    Calculates the Jaro-Winkler similarity score between two strings.

    Args:
        value1 (str): The first string to compare.
        value2 (str): The second string to compare.

    Returns:
        int: The Jaro-Winkler similarity score between the two strings, as an integer between 0 and 100.
    """
    score = int(algs.jaro_winkler.normalized_similarity(value1, value2) * 100)
    return score


def cosine(value1, value2):
    """
    Calculates the cosine similarity (Ochiai coefficient) between two strings
    using token-frequency vectors

    https://en.wikipedia.org/wiki/Cosine_similarity.

    Args:
        value1 (list): The first vector.
        value2 (list): The second vector.

    Returns:
        int: The cosine similarity score between the two vectors, as a percentage.
    """
    score = int(algs.cosine.normalized_similarity(value1, value2) * 100)
    return score


def lcsseq(value1, value2):
    """
    Computes the longest common subsequence (LCS) similarity score between two input strings.

    Args:
        value1 (str): The first input string.
        value2 (str): The second input string.

    Returns:
        int: The LCS similarity score between the two input strings, as a percentage (0-100).
    """
    score = int(algs.lcsseq.normalized_similarity(value1, value2) * 100)
    return score


def lcsstr(value1, value2):
    """
    Calculates the longest common substring (LCS) similarity score between two strings.

    Args:
        value1 (str): The first string to compare.
        value2 (str): The second string to compare.

    Returns:
        int: The LCS similarity score between the two strings, as a percentage (0-100).
    """
    score = int(algs.lcsstr.normalized_similarity(value1, value2) * 100)
    return score


def llm_answer_relevance(response_generator: ResponseGenerator, question, answer):
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
    try:
        result = response_generator.generate_response(
            sys_message=llm_answer_relevance_instruction, prompt=answer
        )
    except Exception as e:
        logger.error(f"Unable to generate answer relevance score: {e}")
        return 0
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    embedding1 = model.encode([str(question)])
    embedding2 = model.encode([str(result)])
    similarity_score = cosine_similarity(embedding1, embedding2)

    return float(similarity_score[0][0] * 100)


def llm_context_precision(response_generator: ResponseGenerator, question, context):
    """
    Verifies whether or not a given context is useful for answering a question.

    Args:
        question (str): The question being asked.
        context (str): The given context.

    Returns:
        int: 1 or 0 depending on if the context is relevant or not.
    """
    prompt = "\nquestion: " + question + "\ncontext: " + context + "\nanswer: "
    try:
        result = response_generator.generate_response(
            sys_message=llm_context_precision_instruction, prompt=prompt
        )
    except Exception as e:
        logger.error(f"Unable to generate context precision score: {e}")
        return 0
    # Since we're only asking for one response, the result is always a boolean 1 or 0
    if "Yes" in result:
        return 100

    return 0


def llm_context_recall(
    response_generator: ResponseGenerator,
    question,
    groundtruth_answer,
    context,
    temperature: int,
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
        context (str): The given context.
        temperature (int): Temperature as defined in the config.

    Returns:
        double: The context recall score generated between the ground truth (expected) and context.
    """
    prompt = (
        "\nquestion: "
        + question
        + "\ncontext: "
        + context
        + "\nanswer: "
        + groundtruth_answer
    )
    result = response_generator.generate_response(
        sys_message=llm_context_recall_instruction,
        prompt=prompt,
        temperature=temperature,
    )
    print(result)
    good_response = '"Attributed": "1"'
    bad_response = '"Attributed": "0"'

    return (
        result.count(good_response)
        / (result.count(good_response) + result.count(bad_response))
    ) * 100


def generate_metrics(experiment_name, run_id, client):
    """
    Generates metrics for a given experiment and run ID.

    Args:
        experiment_name (str): The name of the experiment.
        run_id (int): The ID of the run.
        client (mlflow.MlflowClient): The MLflow client to use for logging the metrics.

    Returns:
        None
    """
    experiment = dict(client.get_experiment_by_name(experiment_name))
    runs_list = client.search_runs([experiment["experiment_id"]])

    models_metrics = {}
    metrics_to_plot = []
    runs_id_to_plot = []

    if len(runs_list) > 0:
        for run in runs_list:
            run_dict = run.to_dictionary()
            single_run_id = run_dict["info"]["run_id"]
            runs_id_to_plot.append(single_run_id)
            if run.data.params.get("run_metrics", {}) != {}:
                metrics = ast.literal_eval(run.data.params["run_metrics"])

                for metric_type, metric_value in metrics.items():
                    if models_metrics.get(metric_type, {}) == {}:
                        metrics_to_plot.append(metric_type)
                        models_metrics[metric_type] = {}

                    models_metrics[metric_type][single_run_id] = metric_value
                logger.debug(models_metrics)
    else:
        current_run = client.get_run(run_id)
        if current_run.data.params.get("run_metrics", {}) != {}:
            metrics = ast.literal_eval(current_run.data.params["run_metrics"])
            for metric_type, metric_value in metrics.items():
                if models_metrics.get(metric_type, {}) == {}:
                    metrics_to_plot.append(metric_type)
                    models_metrics[metric_type] = {}

                models_metrics[metric_type][run_id] = metric_value

    x_axis = []
    y_axis = []

    fig = go.Figure()

    for metric in metrics_to_plot:
        for key, value in models_metrics[metric].items():
            x_axis.append(key)
            y_axis.append(value)

        label = key
        px.line(x_axis, y_axis)
        fig.add_trace(go.Scatter(x=x_axis, y=y_axis, mode="lines+markers", name=label))

        fig.update_layout(
            xaxis_title="run name", yaxis_title=metric, font=dict(size=15)
        )

        plot_name = metric + ".html"
        client.log_figure(run_id, fig, plot_name)

        fig.data = []
        fig.layout = {}
        x_axis = []
        y_axis = []


def draw_hist_df(df, run_id, client):
    """
    Draw a histogram of the given dataframe and log it to the specified run ID.

    Args:
        df (pandas.DataFrame): The dataframe to draw the histogram from.
        run_id (str): The ID of the run to log the histogram to.
        client (mlflow.MlflowClient): The MLflow client to use for logging the histogram.

    Returns:
        None
    """
    fig = px.bar(
        x=df.columns,
        y=df.values.tolist(),
        title="metric comparison",
        color=df.columns,
        labels=dict(x="Metric Type", y="Score", color="Metric Type"),
    )
    plot_name = "all_metrics_current_run.html"
    client.log_figure(run_id, fig, plot_name)


def plot_apk_scores(df, run_id, client):
    fig = px.line(df, x="k", y="score", title="AP@k scores", color="search_type")
    plot_name = "average_precision_at_k.html"
    client.log_figure(run_id, fig, plot_name)


# maybe pull these 2 above and below functions into a single one
def plot_mapk_scores(df, run_id, client):
    fig = px.line(df, x="k", y="map_at_k", title="MAP@k scores", color="search_type")
    plot_name = "mean_average_precision_at_k.html"
    client.log_figure(run_id, fig, plot_name)


def plot_map_scores(df, run_id, client):
    fig = px.bar(df, x="search_type", y="mean", title="MAP scores", color="search_type")
    plot_name = "mean_average_precision_scores.html"
    client.log_figure(run_id, fig, plot_name)


def compute_metrics(
    response_generator: ResponseGenerator,
    question,
    actual,
    expected,
    context,
    metric_type,
):
    """
    Computes a score for the similarity between two strings using a specified metric.

    Args:
        actual (str): The first string to compare.
        expected (str): The second string to compare.
        metric_type (str): The type of metric to use for comparison. Valid options are:
            - "lcsstr": Longest common substring
            - "lcsseq": Longest common subsequence
            - "cosine": Cosine similarity (Ochiai coefficient)
            - "jaro_winkler": Jaro-Winkler distance
            - "hamming": Hamming distance
            - "jaccard": Jaccard similarity
            - "levenshtein": Levenshtein distance
            - "fuzzy": FuzzyWuzzy similarity
            - "bert_all_MiniLM_L6_v2": BERT-based semantic similarity (MiniLM L6 v2 model)
            - "bert_base_nli_mean_tokens": BERT-based semantic similarity (base model, mean tokens)
            - "bert_large_nli_mean_tokens": BERT-based semantic similarity (large model, mean tokens)
            - "bert_large_nli_stsb_mean_tokens": BERT-based semantic similarity (large model, STS-B, mean tokens)
            - "bert_distilbert_base_nli_stsb_mean_tokens": BERT-based semantic similarity (DistilBERT base model, STS-B, mean tokens)
            - "bert_paraphrase_multilingual_MiniLM_L12_v2": BERT-based semantic similarity (multilingual paraphrase model, MiniLM L12 v2)
            - "llm_context_precision": Verifies whether or not a given context is useful for answering a question.
            - "llm_answer_relevance": Scores the relevancy of the answer according to the given question.
            - "llm_context_recall": Scores context recall by estimating TP and FN using annotated answer (ground truth) and retrieved context.
        config (Config): The configuration of the experiment.

    Returns:
        float: The similarity score between the two strings, as determined by the specified metric.
    """
    if metric_type == "lcsstr":
        score = lcsstr(actual, expected)
    elif metric_type == "lcsseq":
        score = lcsseq(actual, expected)
    elif metric_type == "cosine":
        score = cosine(actual, expected)
    elif metric_type == "jaro_winkler":
        score = jaro_winkler(actual, expected)
    elif metric_type == "hamming":
        score = hamming(actual, expected)
    elif metric_type == "jaccard":
        score = jaccard(actual, expected)
    elif metric_type == "levenshtein":
        score = levenshtein(actual, expected)
    elif metric_type == "fuzzy":
        score = fuzzy(actual, expected)
    elif metric_type == "bert_all_MiniLM_L6_v2":
        all_MiniLM_L6_v2 = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        score = compare_semantic_document_values(actual, expected, all_MiniLM_L6_v2)
    elif metric_type == "bert_base_nli_mean_tokens":
        base_nli_mean_tokens = SentenceTransformer(
            "sentence-transformers/bert-base-nli-mean-tokens"
        )
        score = compare_semantic_document_values(actual, expected, base_nli_mean_tokens)
    elif metric_type == "bert_large_nli_mean_tokens":
        large_nli_mean_tokens = SentenceTransformer(
            "sentence-transformers/bert-large-nli-mean-tokens"
        )
        score = compare_semantic_document_values(
            actual, expected, large_nli_mean_tokens
        )
    elif metric_type == "bert_large_nli_stsb_mean_tokens":
        large_nli_stsb_mean_tokens = SentenceTransformer(
            "sentence-transformers/bert-large-nli-stsb-mean-tokens"
        )
        score = compare_semantic_document_values(
            actual, expected, large_nli_stsb_mean_tokens
        )
    elif metric_type == "bert_distilbert_base_nli_stsb_mean_tokens":
        distilbert_base_nli_stsb_mean_tokens = SentenceTransformer(
            "sentence-transformers/distilbert-base-nli-stsb-mean-tokens"
        )
        score = compare_semantic_document_values(
            actual, expected, distilbert_base_nli_stsb_mean_tokens
        )
    elif metric_type == "bert_paraphrase_multilingual_MiniLM_L12_v2":
        paraphrase_multilingual_MiniLM_L12_v2 = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        score = compare_semantic_document_values(
            actual, expected, paraphrase_multilingual_MiniLM_L12_v2
        )
    elif metric_type == "llm_answer_relevance":
        score = llm_answer_relevance(response_generator, actual, expected)
    elif metric_type == "llm_context_precision":
        score = llm_context_precision(response_generator, actual, context)
    elif metric_type == "llm_context_recall":
        score = llm_context_recall(response_generator, question, expected, context)
    else:
        pass

    return score


def evaluate_single_prompt(
    data,
    response_generator,
    metric_types,
    data_list,
    total_precision_scores_by_search_type,
    map_scores_by_search_type,
    average_precision_for_search_type,
):
    actual = remove_spaces(lower(data.actual))
    expected = remove_spaces(lower(data.expected))

    metric_dic = {}

    for metric_type in metric_types:
        score = compute_metrics(
            response_generator,
            data.question,
            actual,
            expected,
            data.context,
            metric_type,
        )
        metric_dic[metric_type] = score
    metric_dic["question"] = data.question
    metric_dic["context"] = data.context
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
    client: mlflow.MlflowClient,
    name_suffix: str,
):
    """
    Evaluates prompts using various metrics and logs the results to MLflow.

    Args:
        environment (Environment): Initialised Environment class containing environment configuration
        config (Config): The configuration settings to use for evaluation.
        index_config (IndexConfig): Parameters of the index such as chunking and embedding model.
        client (mlflow.MlflowClient): The MLflow client to use for logging the results.
        name_suffix (str): Name suffix to use for all outputs created.

    Returns:
        None
    """
    metric_types = config.METRIC_TYPES
    num_search_type = config.SEARCH_VARIANTS
    data_list = []

    pd.set_option("display.max_columns", None)

    total_precision_scores_by_search_type = {}
    map_scores_by_search_type = {}
    average_precision_for_search_type = {}

    handler = QueryOutputHandler(config.QUERY_DATA_LOCATION)

    response_generator = ResponseGenerator(
        environment, config, config.AZURE_OAI_EVAL_DEPLOYMENT_NAME
    )

    query_data_load = handler.load(
        index_config.index_name(), config.EXPERIMENT_NAME, config.JOB_NAME
    )
    question_count = query_data_load[0].question_count

    with ExitStack() as stack:
        executor = stack.enter_context(ThreadPoolExecutor(config.MAX_WORKER_THREADS))
        futures = {
            executor.submit(
                evaluate_single_prompt,
                data,
                response_generator,
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
    columns_to_remove = ["question", "context", "actual", "expected"]
    additional_columns_to_remove = ["search_type"]
    df = pd.DataFrame(data_list)
    df.to_csv(
        os.path.join(config.EVAL_DATA_LOCATION, f"{name_suffix}.csv"), index=False
    )
    logger.debug(f"Eval scores: {df.head()}")

    temp_df = df.drop(columns=columns_to_remove)
    draw_search_chart(temp_df, run_id, client)

    temp_df = temp_df.drop(columns=additional_columns_to_remove)

    if isinstance(num_search_type, str):
        num_search_type = [num_search_type]
    sum_all_columns = temp_df.sum() / (question_count * len(num_search_type))
    sum_df = pd.DataFrame([sum_all_columns], columns=temp_df.columns)

    sum_dict = {}
    for col_name in sum_df.columns:
        sum_dict[col_name] = float(sum_df[col_name].values)

    sum_df.to_csv(
        os.path.join(config.EVAL_DATA_LOCATION, f"sum_{name_suffix}.csv"), index=False
    )

    ap_scores_df = pd.DataFrame(eval_scores_df)
    ap_scores_df.to_csv(
        os.path.join(
            config.EVAL_DATA_LOCATION, f"{name_suffix}_ap_scores_at_k_test.csv"
        ),
        index=False,
    )
    plot_apk_scores(ap_scores_df, run_id, client)
    plot_mapk_scores(ap_scores_df, run_id, client)

    map_scores_df = pd.DataFrame(mean_scores)
    map_scores_df.to_csv(
        os.path.join(config.EVAL_DATA_LOCATION, f"{name_suffix}_map_scores_test.csv"),
        index=False,
    )
    plot_map_scores(map_scores_df, run_id, client)

    common_data = query_data_load[0]
    mlflow.log_param("chunk_size", index_config.chunk_size)
    mlflow.log_param("question_count", common_data.question_count)
    mlflow.log_param("rerank", common_data.rerank)
    mlflow.log_param("rerank_type", common_data.rerank_type)
    mlflow.log_param("crossencoder_model", common_data.crossencoder_model)
    mlflow.log_param("llm_re_rank_threshold", common_data.llm_re_rank_threshold)
    mlflow.log_param("retrieve_num_of_documents", common_data.retrieve_num_of_documents)
    mlflow.log_param("crossencoder_at_k", common_data.crossencoder_at_k)
    mlflow.log_param("chunk_overlap", index_config.overlap)
    mlflow.log_param("embedding_dimension", index_config.embedding_model.dimension)
    mlflow.log_param("embedding_model_name", index_config.embedding_model.name)
    mlflow.log_param("ef_construction", index_config.ef_construction)
    mlflow.log_param("ef_search", index_config.ef_search)
    mlflow.log_param("run_metrics", sum_dict)
    mlflow.log_metrics(sum_dict)
    mlflow.log_artifact(os.path.join(config.EVAL_DATA_LOCATION, f"{name_suffix}.csv"))
    mlflow.log_artifact(
        os.path.join(config.EVAL_DATA_LOCATION, f"sum_{name_suffix}.csv")
    )
    draw_hist_df(sum_df, run_id, client)
    generate_metrics(config.EXPERIMENT_NAME, run_id, client)
    mlflow.end_run()


def draw_search_chart(temp_df, run_id, client):
    """
    Draws a comparison chart of search types across metric types.

    Args:
        temp_df (pandas.DataFrame): The dataframe containing the data to be plotted.
        run_id (int): The ID of the current run.
        client (mlflow.MlflowClient): The MLflow client to use for logging the chart.

    Returns:
        None
    """
    grouped = temp_df.groupby("search_type")
    summed_column = grouped.sum().reset_index()
    fig = sp.make_subplots(rows=len(summed_column.search_type), cols=1)
    for index, row_data in summed_column.iterrows():
        search_type = row_data[0]
        row_data = row_data[1:]
        df = row_data.reset_index(name="metric_value")
        df = df.rename(columns={"index": "metric_type"})
        fig.add_trace(
            go.Bar(
                x=df["metric_type"],
                y=df["metric_value"],
                name=search_type,
                offsetgroup=index,
            ),
            row=1,
            col=1,
        )

        fig.update_xaxes(title_text="Metric type", row=index + 1, col=1)
        fig.update_yaxes(title_text="score", row=index + 1, col=1)
    fig.update_layout(
        font=dict(size=15),
        title_text="Search type comparison across metric types",
        height=4000,
        width=800,
    )
    plot_name = "search_type_current_run.html"
    client.log_figure(run_id, fig, plot_name)
