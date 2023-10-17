
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

from evaluation.spacy_evaluator import SpacyEvaluator

load_dotenv()
warnings.filterwarnings("ignore") 
cli.download("en_core_web_md")
nlp = spacy.load("en_core_web_md")
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
    return text.lower()

def remove_spaces(text):
    return text.strip()


#def compare_rouge(value1, value2):
#    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
#    scores = scorer.score(value1, value2)
#    return scores

def bleu(value1, value2):
    score = sentence_bleu(str.split(value1), value2)
    return score * 100


def fuzzy(doc1, doc2):
    differences = []
    fuzzy_compare_values( doc1, doc2, differences)
    
    return int(sum(differences)/len(differences))

def fuzzy_compare_values(value1, value2, differences):
    similarity_score = fuzz.token_set_ratio(value1, value2)
    differences.append(similarity_score)

def compare_semantic_document_values(doc1, doc2, model_type):
    differences = []
    semantic_compare_values( doc1, doc2, differences, model_type)
    
    return int(sum(differences)/len(differences))

def semantic_compare_values(value1, value2, differences, model_type):
    embedding1 = model_type.encode([str(value1)])
    embedding2 = model_type.encode([str(value2)])
    similarity_score = cosine_similarity(embedding1,embedding2)

    differences.append(similarity_score * 100)

def levenshtein(value1, value2):
    score = int(algs.levenshtein.normalized_similarity(value1, value2) * 100)
    return score

def jaccard(value1, value2):
    score = int(algs.jaccard.normalized_similarity(value1, value2) * 100)
    return score

def hamming(value1, value2):
    score = int(algs.hamming.normalized_similarity(value1, value2) * 100)
    return score

def jaro_winkler(value1, value2):
    score = int(algs.jaro_winkler.normalized_similarity(value1, value2) * 100)
    return score

def cosine(value1, value2):
    score = int(algs.cosine.normalized_similarity(value1, value2) * 100)
    return score

def lcsseq(value1, value2):
    score = int(algs.lcsseq.normalized_similarity(value1, value2) * 100)
    return score

def lcsstr(value1, value2):
    score = int(algs.lcsstr.normalized_similarity(value1, value2) * 100)
    return score

def generate_metrics(experiment_name, run_id):

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
    fig = px.bar(x=df.columns, y = df.values.tolist(),title="metric comparison", color=df.columns,labels=dict(x="Metric Type", y="Score", color="Metric Type"))
    plot_name = "all_metrics_current_run.html"
    client.log_figure(run_id, fig, plot_name)


def compute_metrics(actual, expected, metric_type):
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
    with open('search_config.json', 'r') as json_file:
        data = json.load(json_file)

    metric_types = data["metric_types"]
    data_list = []
    run_name = f"{exp_name}_{formatted_datetime}"
    time.sleep(30)
    mlflow.set_experiment(exp_name)
    mlflow.start_run(run_name=run_name)
    
    run_id = mlflow.active_run().info.run_id
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
    time.sleep(10)


def draw_search_chart(temp_df, run_id):

    grouped = temp_df.groupby('search_type')
    summed_column = grouped.sum().reset_index()  
    num_columns = len(summed_column.columns)
    fig = sp.make_subplots(rows=num_columns + 1, cols= 1)
    for index, row_data in summed_column.iterrows():
        search_type = row_data[0]
        row_data = row_data[1:]
        df = row_data.reset_index(name='metric_value')
        df = df.rename(columns={'index': 'metric_type'})
        fig.add_trace(
            go.Bar(x=df["metric_type"], y=df["metric_value"], name=search_type, ),
            row=index + 1, col=1,
        )

        fig.update_xaxes(title_text='Metric type', row=index + 1, col=1)
        fig.update_yaxes(title_text='score', row=index + 1, col=1)
    fig.update_layout(font=dict(size=15), title_text="Search type comparison across metric types",
                        height=4000, width=800)
    plot_name = "search_type_current_run.html"
    client.log_figure(run_id, fig, plot_name)


def get_recall_score(is_relevant_results: list[bool], total_relevant_docs: int):
    if total_relevant_docs == 0: 
        return 0

    num_of_relevant_docs = is_relevant_results.count(True)
    
    return num_of_relevant_docs/total_relevant_docs

def get_precision_score(is_relevant_results: list[bool]):
    num_of_recommended_docs = len(is_relevant_results)
    if num_of_recommended_docs == 0: 
        return 0
    num_of_relevant_docs = is_relevant_results.count(True)

    return num_of_relevant_docs/num_of_recommended_docs


def evaluate_search_results(search_eval_content, evaluation_content: str, search_relevancy_threshold: float):
    content = []
    search_metrics = []

    evaluator = SpacyEvaluator(similarity_threshold=search_relevancy_threshold)

    
    for eval in search_eval_content:
        recall_scores = []
        precision_scores = []
        total_relevent_docs = []
        is_relevant_results: list[bool] = []

        # get total relevant docs to calulate recall
        for sr in eval['search_results']:
            is_relevant = evaluator.is_relevant(sr["content"], evaluation_content)
            total_relevent_docs.append(is_relevant)

        k = 1
        for sr in eval['search_results']:
            print("++++++++++++++++++++++++++++++++++")
            print(f"Content: {sr['content']}")
            print(f"Search Score: {sr['@search.score']}")

            is_relevant = evaluator.is_relevant(sr["content"], evaluation_content)
            is_relevant_results.append(is_relevant)

            precision_score = get_precision_score(is_relevant_results)
            print(f"Precision Score: {precision_score}@{k}")
            precision_scores.append(f"{precision_score}@{k}")

            recall_score = get_recall_score(is_relevant_results, sum(total_relevent_docs))
            print(f"Recall Score: {recall_score}@{k}")
            recall_scores.append(f"{recall_score}@{k}")

            # TODO: should we only append content when it is relevant?
            content.append(sr['content']) 
            k += 1

        metric = {
            "question": eval['question'],
            "recall_scores": recall_scores,
            "precision_scores": precision_scores,
        }
        search_metrics.append(metric)

    result = {
        "content": content,
        'search_metrics': search_metrics
    }  

    return result