import os
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from umap import UMAP
from scipy.spatial.distance import cdist
from rag_experiment_accelerator.checkpoint import cache_with_checkpoint
from rag_experiment_accelerator.config.config import Config
from rag_experiment_accelerator.utils.logging import get_logger
import shutil

matplotlib.use("Agg")
plt.style.use("ggplot")
warnings.filterwarnings("ignore")
logger = get_logger(__name__)


def load_parser():
    from spacy import load

    try:
        parser = load("en_core_web_lg", disable=["ner"])
    except OSError:
        logger.info("Downloading spacy language model: en_core_web_lg")
        from spacy.cli import download

        download("en_core_web_lg")
        parser = load("en_core_web_lg", disable=["ner"])

    parser.max_length = 7000000

    return parser


def spacy_tokenizer(sentence, parser):
    """
    Tokenizes a sentence using the Spacy library.

    Args:
        sentence (str): The input sentence to be tokenized.

    Returns:
        str: The tokenized sentence.

    """

    if not isinstance(sentence, str):
        sentence = sentence["content"]

    tokens = [
        word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_
        for word in parser(sentence)
        if not word.is_stop and not word.is_punct
    ]
    tokenized_sentence = " ".join([token for token in tokens])
    return tokenized_sentence


def determine_optimum_k_elbow(embeddings_2d, X, min_cluster, max_cluster, result_dir):
    """
    Determines the optimal number of clusters using the Elbow Method.

    Args:
        embeddings_2d (numpy.ndarray): 2D embeddings of the data.
        X (numpy.ndarray): Input data.
        min_cluster (int): Minimum number of clusters to consider.
        max_cluster (int): Maximum number of clusters to consider.
        result_dir (str): Directory to save the output files.

    Returns:
        int: The optimum number of clusters.

    """
    logger.info("Determining optimal k")

    # Run kmeans with many different k
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}

    K = range(min_cluster, max_cluster)
    for k in tqdm(K):
        k_means = KMeans(n_clusters=k, random_state=42).fit(embeddings_2d)
        k_means.fit(embeddings_2d)
        distortions.append(
            sum(
                np.min(
                    cdist(embeddings_2d, k_means.cluster_centers_, "euclidean"), axis=1
                )
            )
            / X.shape[0]
        )
        inertias.append(k_means.inertia_)
        mapping1[k] = (
            sum(
                np.min(
                    cdist(embeddings_2d, k_means.cluster_centers_, "euclidean"), axis=1
                )
            )
            / X.shape[0]
        )
        mapping2[k] = k_means.inertia_

    lbig = max(abs(x - y) for (x, y) in zip(distortions[1:], distortions[:-1]))

    x_line = [K[0], K[-1]]
    y_line = [distortions[0], distortions[-1]]

    opt_k = []

    for i, distortion in enumerate(distortions):
        if i != len(distortions) - 1:
            if distortion - distortions[i + 1] == lbig:
                opt_k.append(
                    list(mapping1.keys())[
                        list(mapping1.values()).index(distortions[i + 3])
                    ]
                )

    logger.info(f"The optimum cluster number is {opt_k[0]}")
    optimum_k = opt_k[0]

    plt.plot(K, distortions, "b-")
    plt.plot(x_line, y_line, "r")
    plt.xlabel("k")
    plt.ylabel("Distortion")
    plt.title("The Elbow Method showing the optimal k")
    plt.savefig(f"{result_dir}/elbow_{optimum_k}.png")

    return optimum_k


def vectorize_tfidf(text, max_features):
    """
    Vectorizes the given text using TF-IDF representation.

    Args:
        text (list): A list of strings representing the text documents.
        max_features (int): The maximum number of features to keep.

    Returns:
        scipy.sparse.csr_matrix: The TF-IDF matrix representation of the text.
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(text)
    return X


def dataframe_to_chunk_dict(df_concat):
    """
    Convert a dataframe to a dictionary of chunks.

    Args:
        df_concat (pandas.DataFrame): The dataframe containing the chunks.

    Returns:
        list: A list of dictionaries, where each dictionary represents a chunk with its corresponding text.
    """
    sampled_chunks = []
    for i, row in enumerate(df_concat.itertuples()):
        chunk = {row.chunk: row.text}
        sampled_chunks.append(chunk)

    return sampled_chunks


def chunk_dict_to_dataframe(all_chunks):
    """
    Convert a list of dictionaries containing chunks and text into a pandas DataFrame.

    Parameters:
    all_chunks (list[dict]): A list of dictionaries where each dictionary contains a chunk and its corresponding text.

    Returns:
    df (pandas.DataFrame): A DataFrame with three columns - 'chunk', 'text' and 'filename, where 'chunk' contains the chunks and 'text' contains the corresponding text and 'filename' the file name.
    """

    chunks = []
    text = []
    filename = []

    for row in all_chunks:
        key, value = list(row.items())[0]
        chunks.append(key)
        text.append(value)
        filename.append(value["metadata"]["source"])

    df = pd.DataFrame({"chunk": chunks, "text": text, "filename": filename})

    return df


def cluster_kmeans(embeddings_2d, optimum_k, df, result_dir):
    """
    Perform K-means clustering on 2D embeddings.

    Args:
        embeddings_2d (numpy.ndarray): 2D embeddings array.
        optimum_k (int): Number of clusters to create.
        df (pandas.DataFrame): DataFrame containing additional data.
        result_dir (str): Directory to save the clustering results.

    Returns:
        tuple: A tuple containing the following lists:
            - x (list): X-coordinates of the embeddings.
            - y (list): Y-coordinates of the embeddings.
            - text (list): Text data from the DataFrame.
            - processed_text (list): Processed text data from the DataFrame.
            - chunk (list): Chunk data from the DataFrame.
            - prediction (list): Cluster labels assigned by K-means.
            - prediction_values (list): Unique cluster labels.
            - filenames (list): File names of the sampled data.

    """
    logger.info("Clustering chunks")
    kmeans = KMeans(n_clusters=optimum_k)
    kmeans.fit(embeddings_2d)

    # Plot
    fig = px.scatter(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], color=kmeans.labels_)
    fig.write_image(
        f"{result_dir}/all_cluster_predictions_cluster_number_{optimum_k}.jpg"
    )

    # Save
    filenames = (
        x
    ) = y = text = processed_text = chunk = prediction = prediction_values = []
    x = embeddings_2d[:, 0].tolist()
    y = embeddings_2d[:, 1].tolist()
    text = df["text"].tolist()
    processed_text = df["processed_text"].tolist()
    chunk = df["chunk"].tolist()
    prediction = kmeans.labels_.tolist()
    prediction_values = list(set(kmeans.labels_.tolist()))
    filenames = list(set(df["filename"].tolist()))

    return x, y, text, processed_text, chunk, prediction, prediction_values, filenames


@cache_with_checkpoint(id="index_name")
def cluster(index_name, all_chunks, config: Config, parser):
    """
    Clusters the given chunks of documents using TF-IDF and K-means clustering.

    Args:
        index_name (str): The name of the index - used only for uniqueness as checkpoint key.
        all_chunks (list): A list of document chunks.
        config (object): The configuration object.

    Returns:
        dict: A dictionary containing the sampled document chunks.

    """
    logger.info(f"Sampling - Original Document chunk length {len(all_chunks)}")
    df = chunk_dict_to_dataframe(all_chunks)

    # Tokenize and remove punctuation and stop words
    tqdm.pandas()
    df["processed_text"] = df["text"].progress_apply(
        lambda text: spacy_tokenizer(text, parser)
    )

    # Run TF-IDF
    logger.info("Run TF-IDF")
    text = df["processed_text"].values
    max_features = 2**12
    X = vectorize_tfidf(text, max_features)

    logger.info("Reducing Umap")
    reducer = UMAP()
    embeddings_2d = reducer.fit_transform(X)

    if config.index.sampling.optimum_k == "auto":
        optimum_k = determine_optimum_k_elbow(
            embeddings_2d,
            X,
            config.index.sampling.min_cluster,
            config.index.sampling.max_cluster,
            config.path.sampling_output_dir,
        )
    else:
        optimum_k = config.index.sampling.optimum_k

    # Cluster

    (
        x,
        y,
        text,
        processed_text,
        chunk,
        prediction,
        prediction_values,
        filenames,
    ) = cluster_kmeans(embeddings_2d, optimum_k, df, config.sampling_output_dir)

    # Capture all predictions
    data = {"x": x, "y": y, "text": text, "prediction": prediction, "chunk": chunk}
    df = pd.DataFrame(data)
    df.to_csv(
        f"{config.path.sampling_output_dir}/all_cluster_predictions_cluster_number_{config.index.sampling.optimum_k}.csv",
        sep=",",
    )

    # Sample the clusters as dataframes
    g = globals()
    for i in prediction_values:
        g["l_{0}".format(i)] = df[df["prediction"] == i]

        if len(g["l_{0}".format(i)]) > round(
            (len(df) * (config.index.sampling.percentage / 100))
            / len(prediction_values)
        ):
            g["l_{0}".format(i)] = g["l_{0}".format(i)].sample(
                n=round(
                    (len(df) * (config.index.sampling.percentage / 100))
                    / len(prediction_values)
                ),
                random_state=42,
            )

    df_list = [g["l_{0}".format(i)] for i in prediction_values]

    # Concatenate the list of DataFrames into a single DataFrame
    df_concat = pd.concat(df_list)
    df_concat.to_csv(
        config.path.sampled_cluster_predictions_path(config.index.sampling.optimum_k),
        sep=",",
    )
    # Rebuild sampled chunks dict
    sampled_chunks = dataframe_to_chunk_dict(df_concat)
    logger.info(f"Sampled Document chunk length {len(sampled_chunks)}")

    # Preserve the sampled files into directory
    for filename in filenames:
        try:
            fn = os.path.basename(filename)
            os.makedirs(
                config.sampling_output_dir + "/" + config.JOB_NAME, exist_ok=True
            )
            shutil.copy2(
                filename, config.sampling_output_dir + "/" + config.JOB_NAME + "/" + fn
            )
        except OSError as e:
            logger.info(f"file {filename} could not be copied with metadata {e}")
            continue
    logger.info(
        f"Sampled Documents have been copied to {config.sampling_output_dir + '/' + config.JOB_NAME + '/'}"
    )

    return sampled_chunks
