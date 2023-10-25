
from azure.ai.ml.entities import Data
from azure.ai.ml import Input
from dotenv import load_dotenv
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate import meteor
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
import pandas as pd
import azure.ai.ml._artifacts._artifact_utilities as artifact_utils
import warnings
from rouge_score import rouge_scorer
import json
import ast
import time
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from datetime import datetime
import spacy
import textdistance
import mlflow
from mlflow import MlflowClient
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import os
from spacy import cli
from numpy import mean


warnings.filterwarnings("ignore") 

nlp = spacy.load("en_core_web_lg")
current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")
algs = textdistance.algorithms

pd.set_option('display.max_columns', None)

all_MiniLM_L6_v2 = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
base_nli_mean_tokens = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
large_nli_mean_tokens = SentenceTransformer('sentence-transformers/bert-large-nli-mean-tokens')
large_nli_stsb_mean_tokens = SentenceTransformer('sentence-transformers/bert-large-nli-stsb-mean-tokens')
distilbert_base_nli_stsb_mean_tokens = SentenceTransformer('sentence-transformers/distilbert-base-nli-stsb-mean-tokens')
paraphrase_multilingual_MiniLM_L12_v2 = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

ml_client = MLClient(
    DefaultAzureCredential(), os.environ['SUBSCRIPTION_ID'],os.environ['RESOURCE_GROUP_NAME'], os.environ['WORKSPACE_NAME']
)
mlflow_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
mlflow.set_tracking_uri(mlflow_tracking_uri)
client = MlflowClient(mlflow_tracking_uri)

if not os.path.exists("./eval_score"):
    os.makedirs("./eval_score")

def process_text(text):
    """
    Processes the given text by removing stop words, punctuation, and lemmatizing the remaining words.

    Args:
        text (str): The text to process.

    Returns:
        str: The processed text.
    """
    doc = nlp(str(text))
    result = []
    for token in doc:
        if token.text in nlp.Defaults.stop_words:
            continue
        if token.is_punct:
            continue
        if token.lemma_ == '-PRON-':
            continue
        result.append(token.lemma_)
    return " ".join(result)

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


#def compare_rouge(value1, value2):
#    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
#    scores = scorer.score(value1, value2)
#    return scores

def bleu(value1, value2):
    score = sentence_bleu(str.split(value1), value2)
    return score * 100


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
    fuzzy_compare_values( doc1, doc2, differences)
    
    return int(sum(differences)/len(differences))

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
        model_type (str): The type of model to use for comparison.

    Returns:
        int: The percentage of differences between the two documents.
    """
    differences = []
    semantic_compare_values(doc1, doc2, differences, model_type)
    
    return int(sum(differences)/len(differences))

from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def semantic_compare_values(value1: str, value2: str, differences: List[float], model_type: SentenceTransformer) -> None:
    """
    Computes the semantic similarity score between two values using a pre-trained SentenceTransformer model.

    Args:
        value1 (str): The first value to compare.
        value2 (str): The second value to compare.
        differences (List[float]): A list to store the similarity score between the two values.
        model_type (SentenceTransformer): A pre-trained SentenceTransformer model to encode the values.

    Returns:
        None
    """
    embedding1 = model_type.encode([str(value1)])
    embedding2 = model_type.encode([str(value2)])
    similarity_score = cosine_similarity(embedding1, embedding2)

    differences.append(similarity_score * 100)

    
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def semantic_compare_values(value1: str, value2: str, differences: List[float], model_type: SentenceTransformer) -> None:
    """
    Computes the semantic similarity score between two values using the provided SentenceTransformer model.

    Args:
        value1 (str): The first value to compare.
        value2 (str): The second value to compare.
        differences (List[float]): A list to append the similarity score to.
        model_type (SentenceTransformer): The SentenceTransformer model to use for encoding the values.

    Returns:
        None
    """
    embedding1 = model_type.encode([str(value1)])
    embedding2 = model_type.encode([str(value2)])
    similarity_score = cosine_similarity(embedding1, embedding2)

    differences.append(similarity_score * 100)
    
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def semantic_compare_values(value1: str, value2: str, differences: List[float], model_type: SentenceTransformer) -> None:
    """
    Computes the semantic similarity between two values using a pre-trained SentenceTransformer model.

    Args:
        value1 (str): The first value to compare.
        value2 (str): The second value to compare.
        differences (List[float]): A list to store the similarity scores.
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
    Calculates the cosine similarity between two vectors.

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

import ast
import plotly.express as px
import plotly.graph_objs as go

def generate_metrics(experiment_name, run_id):
    """
    Generates metrics for a given experiment and run ID.

    Args:
        experiment_name (str): The name of the experiment.
        run_id (int): The ID of the run.

    Returns:
        None
    """
    experiment = dict(client.get_experiment_by_name(experiment_name))
    runs_list = client.search_runs([experiment['experiment_id']])
 
    models_metrics = {}
    metrics_to_plot = []
    runs_id_to_plot = []

    if len(runs_list) > 0:
        for run in runs_list:
            run_dict = run.to_dictionary()
            single_run_id = run_dict['info']['run_id']
            runs_id_to_plot.append(single_run_id)
            if run.data.params.get("run_metrics", {}) != {}:
                metrics = ast.literal_eval(run.data.params['run_metrics'])

                for metric_type, metric_value in metrics.items():
                    if models_metrics.get(metric_type, {}) == {}:
                        metrics_to_plot.append(metric_type)
                        models_metrics[metric_type] = {}
                        models_metrics[metric_type][single_run_id] = metric_value
                    else:
                        models_metrics[metric_type][single_run_id]= metric_value
                print(models_metrics)
    else:
        current_run = client.get_run(run_id)
        if run.data.params.get("run_metrics", {}) != {}:
            metrics = ast.literal_eval(current_run.data.params['run_metrics'])
            for metric_type, metric_value in metrics.items():
                if models_metrics.get(metric_type, {}) == {}:
                    metrics_to_plot.append(metric_type)
                    models_metrics[metric_type] = {}
                    models_metrics[metric_type][run_id] = metric_value
                else:
                    models_metrics[metric_type][run_id]= metric_value
                
    x_axis = []
    y_axis = []

    fig = go.Figure()

    for metric in metrics_to_plot:
        for key, value in models_metrics[metric].items():
            x_axis.append(key) 
            y_axis.append(value) 

        label = key
        px.line(x_axis, y_axis, )
        fig.add_trace(go.Scatter(x=x_axis,
                                    y=y_axis,
                                    mode='lines+markers',
                                    name=label)
                                )
        
        fig.update_layout(xaxis_title='run name',
                        yaxis_title=metric,
                        font=dict(size=15)
                        )

        plot_name = metric + ".html"
        client.log_figure(run_id, fig, plot_name)

        fig.data = []
        fig.layout = {}
        x_axis = []
        y_axis = []

def draw_hist_df(df, run_id):
    """
    Draw a histogram of the given dataframe and log it to the specified run ID.

    Args:
        df (pandas.DataFrame): The dataframe to draw the histogram from.
        run_id (str): The ID of the run to log the histogram to.

    Returns:
        None
    """
    fig = px.bar(x=df.columns, y = df.values.tolist(),title="metric comparison", color=df.columns,labels=dict(x="Metric Type", y="Score", color="Metric Type"))
    plot_name = "all_metrics_current_run.html"
    client.log_figure(run_id, fig, plot_name)

def plot_apk_scores(df, run_id, client):
    fig = px.line(df, x="k", y="score",title="AP@k scores", color="search_type")
    plot_name = "average_precision_at_k.html"
    client.log_figure(run_id, fig, plot_name)

# maybe pull these 2 above and below functions into a single one
def plot_mapk_scores(df, run_id, client):
    fig = px.line(df, x="k", y="map_at_k",title="MAP@k scores", color="search_type")
    plot_name = "mean_average_precision_at_k.html"
    client.log_figure(run_id, fig, plot_name)

def plot_map_scores(df, run_id, client):
    fig = px.bar(df, x="search_type", y="mean",title="MAP scores", color="search_type")
    plot_name = "mean_average_precision_scores.html"
    client.log_figure(run_id, fig, plot_name)

def compute_metrics(actual, expected, metric_type):
    """
    Computes a score for the similarity between two strings using a specified metric.

    Args:
        actual (str): The first string to compare.
        expected (str): The second string to compare.
        metric_type (str): The type of metric to use for comparison. Valid options are:
            - "lcsstr": Longest common substring
            - "lcsseq": Longest common subsequence
            - "cosine": Cosine similarity
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
        score = compare_semantic_document_values(actual, expected,all_MiniLM_L6_v2 )
    elif metric_type == "bert_base_nli_mean_tokens":
        score = compare_semantic_document_values(actual, expected,base_nli_mean_tokens )
    elif metric_type == "bert_large_nli_mean_tokens":
        score = compare_semantic_document_values(actual, expected,large_nli_mean_tokens )
    elif metric_type == "bert_large_nli_stsb_mean_tokens":
        score = compare_semantic_document_values(actual, expected,large_nli_stsb_mean_tokens )
    elif metric_type == "bert_distilbert_base_nli_stsb_mean_tokens":
        score = compare_semantic_document_values(actual, expected,distilbert_base_nli_stsb_mean_tokens )
    elif metric_type == "bert_paraphrase_multilingual_MiniLM_L12_v2":
        score = compare_semantic_document_values(actual, expected,paraphrase_multilingual_MiniLM_L12_v2 )
    else:
        pass

    return score

def evaluate_prompts(exp_name, data_path, chunk_size, chunk_overlap, embedding_dimension, efConstruction, efsearch):
    """
    Evaluates prompts using various metrics and logs the results to MLflow.

    Args:
        exp_name (str): Name of the experiment to log the results to.
        data_path (str): Path to the file containing the prompts to evaluate.
        chunk_size (int): Size of the chunks to split the prompts into.
        chunk_overlap (int): Amount of overlap between the chunks.
        embedding_dimension (int): Dimension of the embeddings to use.
        efConstruction (int): Number of trees to use during index construction.
        efsearch (int): Number of trees to use during search.

    Returns:
        None
    """

    with open('search_config.json', 'r') as json_file:
        data = json.load(json_file)

    metric_types = data["metric_types"]
    data_list = []
    run_name = f"{exp_name}_{formatted_datetime}"
    # time.sleep(30)
    mlflow.set_experiment(exp_name)
    mlflow.start_run(run_name=run_name)
    
    run_id = mlflow.active_run().info.run_id

    total_precision_scores_by_search_type = {}
    map_scores_by_search_type = {}
    average_precision_for_search_type = {}
    with open(data_path, 'r') as file:
        for line in file: 
            data = json.loads(line)
            actual = data.get("actual")
            expected = data.get("expected")
            search_type = data.get("search_type")
            rerank = data.get("rerank")
            rerank_type = data.get("rerank_type")
            crossencoder_model = data.get("crossencoder_model")
            llm_re_rank_threshold = data.get("llm_re_rank_threshold")
            retrieve_num_of_documents = data.get("retrieve_num_of_documents")
            cross_encoder_at_k = data.get("cross_encoder_at_k")
            question_count = data.get("question_count")
            search_evals = data.get("search_evals")
            
            actual = remove_spaces(lower(actual))
            expected = remove_spaces(lower(expected))

            metric_dic = {}

            for metric_type in metric_types:
                score = compute_metrics(actual, expected, metric_type)
                metric_dic[metric_type] = score
            metric_dic["actual"] = actual
            metric_dic["expected"] = expected
            metric_dic["search_type"] = search_type
            data_list.append(metric_dic)

            if not total_precision_scores_by_search_type.get(search_type):
                total_precision_scores_by_search_type[search_type] = {}
                map_scores_by_search_type[search_type] = []
                average_precision_for_search_type[search_type] = []
            for eval in search_evals:
                scores = eval.get('precision_scores')
                if scores:
                    average_precision_for_search_type[search_type].append(mean(scores))
                for i, score in enumerate(scores):
                    if total_precision_scores_by_search_type[search_type].get(i+1):
                        total_precision_scores_by_search_type[search_type][i+1].append(score)
                    else:
                        total_precision_scores_by_search_type[search_type][i+1] = [score]

    eval_scores_df = {
        "search_type": [],
        "k": [],
        "score": [],
        "map_at_k": []
    }

    for search_type, scores_at_k in total_precision_scores_by_search_type.items():
        for k, scores in scores_at_k.items():
            avg_at_k = mean(scores)
            # not sure if this would be problematic or not.
            eval_scores_df['search_type'].append(search_type)
            eval_scores_df['k'].append(k)
            eval_scores_df['score'].append(avg_at_k)
            mean_at_k = mean(eval_scores_df['score'][:k])
            eval_scores_df['map_at_k'].append(mean_at_k)

    mean_scores = {
        "search_type": [],
        "mean": []
    }

    for search_type, scores in average_precision_for_search_type.items():
        mean_scores['search_type'].append(search_type)
        mean_scores['mean'].append(mean(scores))
    
    run_id = mlflow.active_run().info.run_id
    columns_to_remove = ['actual', 'expected']
    additional_columns_to_remove = ['search_type']
    df = pd.DataFrame(data_list)
    df.to_csv(f"eval_score/{formatted_datetime}.csv", index=False)
    print(df.head())
    
    temp_df = df.drop(columns=columns_to_remove)
    draw_search_chart(temp_df, run_id)
    
    temp_df = temp_df.drop(columns=additional_columns_to_remove)
    
    sum_all_columns = temp_df.sum() / question_count
    sum_df = pd.DataFrame([sum_all_columns], columns=temp_df.columns)
    
    sum_dict = {}
    for col_name in sum_df.columns:
        sum_dict[col_name] = float(sum_df[col_name].values)

    sum_df.to_csv(f"eval_score/sum_{formatted_datetime}.csv", index=False)

    ap_scores_df = pd.DataFrame(eval_scores_df)
    ap_scores_df.to_csv(f"eval_score/{formatted_datetime}_ap_scores_at_k_test.csv", index=False)
    plot_apk_scores(ap_scores_df, run_id, client)
    plot_mapk_scores(ap_scores_df, run_id, client)

    map_scores_df = pd.DataFrame(mean_scores)
    map_scores_df.to_csv(f"eval_score/{formatted_datetime}_map_scores_test.csv", index=False)
    plot_map_scores(map_scores_df, run_id, client)

    mlflow.log_param("question_count",question_count )
    mlflow.log_param("rerank",rerank )
    mlflow.log_param("rerank_type",rerank_type )
    mlflow.log_param("crossencoder_model",crossencoder_model )
    mlflow.log_param("llm_re_rank_threshold",llm_re_rank_threshold )
    mlflow.log_param("retrieve_num_of_documents",retrieve_num_of_documents )
    mlflow.log_param("cross_encoder_at_k",cross_encoder_at_k )
    mlflow.log_param("chunk_overlap",chunk_overlap )
    mlflow.log_param("embedding_dimension",embedding_dimension )
    mlflow.log_param("efConstruction",efConstruction )
    mlflow.log_param("efsearch",efsearch )
    mlflow.log_param("run_metrics",sum_dict )
    mlflow.log_metrics(sum_dict)
    mlflow.log_artifact(f"eval_score/{formatted_datetime}.csv")
    mlflow.log_artifact(f"eval_score/sum_{formatted_datetime}.csv")
    draw_hist_df(sum_df,run_id)
    generate_metrics(exp_name, run_id)
    mlflow.end_run()
    # time.sleep(10)
def draw_search_chart(temp_df, run_id):
    """
    Draws a comparison chart of search types across metric types.

    Args:
        temp_df (pandas.DataFrame): The dataframe containing the data to be plotted.
        run_id (int): The ID of the current run.

    Returns:
        None
    """
    grouped = temp_df.groupby('search_type')
    summed_column = grouped.sum().reset_index()  
    fig = sp.make_subplots(rows=len(summed_column.search_type), cols= 1)
    for index, row_data in summed_column.iterrows():
        search_type = row_data[0]
        row_data = row_data[1:]
        df = row_data.reset_index(name='metric_value')
        df = df.rename(columns={'index': 'metric_type'})
        fig.add_trace(
            go.Bar(x=df["metric_type"], y=df["metric_value"], name=search_type, offsetgroup=index),
            row=1, col=1,
        )

        fig.update_xaxes(title_text='Metric type', row=index + 1, col=1)
        fig.update_yaxes(title_text='score', row=index + 1, col=1)
    fig.update_layout(font=dict(size=15), title_text="Search type comparison across metric types",
                        height=4000, width=800)
    plot_name = "search_type_current_run.html"
    client.log_figure(run_id, fig, plot_name)

