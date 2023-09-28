
from azure.ai.ml.entities import Data
from azure.ai.ml import Input

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
import os

warnings.filterwarnings("ignore") 

nlp = spacy.load("en_core_web_lg")
current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")
algs = textdistance.algorithms

pd.set_option('display.max_columns', None)

all_MiniLM_L6_v2 = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
bert_base_nli_mean_tokens = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
bert_large_nli_mean_tokens = SentenceTransformer('sentence-transformers/bert-large-nli-mean-tokens')
bert_large_nli_stsb_mean_tokens = SentenceTransformer('sentence-transformers/bert-large-nli-stsb-mean-tokens')
distilbert_base_nli_stsb_mean_tokens = SentenceTransformer('sentence-transformers/distilbert-base-nli-stsb-mean-tokens')
paraphrase_multilingual_MiniLM_L12_v2 = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

ml_client = MLClient(
    DefaultAzureCredential(), os.environ['SUBSCRIPTION_ID'],os.environ['RESOURCE_GROUP_NAME'], os.environ['WORKSPACE_NAME']
)
mlflow_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri
mlflow.set_tracking_uri(mlflow_tracking_uri)
client = MlflowClient(mlflow_tracking_uri)

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

def compare_sentence_bleu(value1, value2):
    score = sentence_bleu(str.split(value1), value2)
    return score * 100


def compare_fuzzy_document_values(doc1, doc2):
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

def generate_metrics(experiment_name, run_id):

    experiment = dict(client.get_experiment_by_name(experiment_name))
    runs_list = client.search_runs([experiment['experiment_id']], filter_string="attributes.status = 'Finished'")
 
    models_metrics = {}
    metrics_to_plot = []
    runs_id_to_plot = []

    for run in runs_list:
        run_dict = run.to_dictionary()
        single_run_id = run_dict['info']['run_id']
        runs_id_to_plot.append(single_run_id)
        metrics = ast.literal_eval(run.data.params['run_metrics'])

        for metric_type, metric_value in metrics.items():
            if models_metrics.get(metric_type, {}) == {}:
                metrics_to_plot.append(metric_type)
                models_metrics[metric_type] = {}
                models_metrics[metric_type][single_run_id] = metric_value
            else:
                models_metrics[metric_type][single_run_id]= metric_value
        print(models_metrics)
                
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
    fig = px.bar(x=df.columns, y = df.values.tolist())
    plot_name = "metrics.html"
    client.log_figure(run_id, fig, plot_name)

data_list = []

def evaluate_prompts(exp_name, data_path, chunk_size, chunk_overlap, embedding_dimension, efConstruction, efsearch):
    run_name = f"{exp_name}_{formatted_datetime}"
    mlflow.set_experiment(exp_name)
    mlflow.start_run(run_name=run_name)

    with open(data_path, 'r') as file:
        for line in file: 
            if len(line) < 5:
                continue
            data = json.loads(line)
            actual = data.get("actual")
            expected = data.get("expected")
            search_type = data.get("search_type")

            actual = remove_spaces(lower(actual))
            expected = remove_spaces(lower(expected))

            # distance based evaluation
            leve = int(algs.levenshtein.normalized_similarity(actual, expected) * 100)
            jaccard = int(algs.jaccard.normalized_similarity(actual, expected) * 100)
            hamming = int(algs.hamming.normalized_similarity(actual, expected) * 100)
            jaro_w = int(algs.jaro_winkler.normalized_similarity(actual, expected) * 100)

            # token based evaluation
            fuzz_s = compare_fuzzy_document_values(actual, expected)
            cosine = int(algs.cosine.normalized_similarity(actual, expected) * 100)

            # sequence based evaluation
            lcsseq = int(algs.lcsseq.normalized_similarity(actual, expected) * 100)
            lcsstr = int(algs.lcsstr.normalized_similarity(actual, expected) * 100)
            
            # semantic similarity evaluation
            all_MiniLM_L6_v2_output = compare_semantic_document_values(actual, expected,all_MiniLM_L6_v2 )
            bert_base_nli_mean_tokens_output = compare_semantic_document_values(actual, expected,bert_base_nli_mean_tokens )
            bert_large_nli_mean_tokens_output = compare_semantic_document_values(actual, expected,bert_large_nli_mean_tokens )
            bert_large_nli_stsb_mean_tokens_output = compare_semantic_document_values(actual, expected,bert_large_nli_stsb_mean_tokens )
            distilbert_base_nli_stsb_mean_tokens_output = compare_semantic_document_values(actual, expected,distilbert_base_nli_stsb_mean_tokens )
            paraphrase_multilingual_MiniLM_L12_v2_output = compare_semantic_document_values(actual, expected,paraphrase_multilingual_MiniLM_L12_v2 )

            bleu = compare_sentence_bleu(actual, expected)
            data_list.append({"bleu": bleu, "fuzzy":fuzz_s,"leve": leve,"jaccard":jaccard,"hamming":hamming,"jaro_w":jaro_w,"cosine":cosine,"lcsseq":lcsseq,"lcsstr":lcsstr,"MiniLM":all_MiniLM_L6_v2_output,"bert_base":bert_base_nli_mean_tokens_output,"bert_large":bert_large_nli_mean_tokens_output,"bert_large_stsb":bert_large_nli_stsb_mean_tokens_output,"distilbert_nli_stsb":distilbert_base_nli_stsb_mean_tokens_output,"multilingual_MiniLM":paraphrase_multilingual_MiniLM_L12_v2_output,"actual":actual, "expected":expected, "search_type": search_type})

    columns_to_remove = ['actual', 'expected', 'search_type']
    df = pd.DataFrame(data_list)
    print(df.head(20))
    
    temp_df = df.drop(columns=columns_to_remove)
    
    sum_all_columns = temp_df.sum()
    sum_df = pd.DataFrame([sum_all_columns], columns=temp_df.columns)
    
    sum_dict = {}
    for col_name in sum_df.columns:
        sum_dict[col_name] = float(sum_df[col_name].values)

    sum_df.to_csv(f"eval_score/sum_{formatted_datetime}.csv", index=False)
    df.to_csv(f"eval_score/{formatted_datetime}.csv", index=False)
    mlflow.log_param("chunk_size",chunk_size )
    mlflow.log_param("chunk_overlap",chunk_overlap )
    mlflow.log_param("embedding_dimension",embedding_dimension )
    mlflow.log_param("efConstruction",efConstruction )
    mlflow.log_param("efsearch",efsearch )
    mlflow.log_param("run_metrics",sum_dict )
    mlflow.log_metrics(sum_dict)
    mlflow.log_artifact(f"eval_score/{formatted_datetime}.csv")
    mlflow.log_artifact(f"eval_score/sum_{formatted_datetime}.csv")
    run_id = mlflow.active_run().info.run_id
    
    draw_hist_df(sum_df,run_id)
    generate_metrics(exp_name, run_id)
    mlflow.end_run()




