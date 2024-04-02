from sklearn.metrics.pairwise import cosine_similarity
from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


def get_semantic_similarity(embeddings_dict, text_dict, threshold):
    high_similarity = {}
    low_similarity = {}

    embeds = generate_pairwise_embeddings(embeddings=embeddings_dict, threshold=threshold)
    low_sim, high_sim = get_similar_vectors(embeds, text_dict, threshold)

    if low_sim:
        low_similarity.update(low_sim)

    for key, array_of_keys in high_sim.items():
        similar_text = ""
        for key_dep in array_of_keys:
            similar_text += text_dict[key_dep]["content"] + ", "
        similar_text += ", " + text_dict[key]["content"]
        high_similarity[key] = {"content": similar_text, "metadata": text_dict[key]["metadata"]}

    return high_similarity, low_similarity


def generate_pairwise_embeddings(embeddings, threshold):
    cosine_similarities = {}

    for key1, embedding1 in embeddings.items():
        for key2, embedding2 in embeddings.items():
            if key1 != key2:
                similarity = cosine_similarity([embedding1], [embedding2])[0][0]
                logger.debug(f"{similarity}--{threshold}")
                key_to_check = (key1, key2)

                if key_to_check in cosine_similarities or tuple(reversed(key_to_check)) in cosine_similarities:
                    pass
                else:
                    cosine_similarities[(key1, key2)] = similarity

    return cosine_similarities


def get_similar_vectors(embeddings, text_dict, threshold):
    below_threshold_dict = {}
    above_threshold_dict = {}
    temp_set = set()

    for key, value in embeddings.items():
        key1, key2 = key

        if float(value) < float(threshold):
            if key1 not in temp_set:
                temp_set.add(key1)
                below_threshold_dict[key1] = text_dict[key1]

            if key2 not in temp_set:
                temp_set.add(key2)
                below_threshold_dict[key2] = text_dict[key2]

        else:
            if key1 not in above_threshold_dict:
                above_threshold_dict[key1] = []

            above_threshold_dict[key1].append(key2)

            if key1 in below_threshold_dict:
                below_threshold_dict.pop(key1)
            if key2 in below_threshold_dict:
                below_threshold_dict.pop(key2)

    return below_threshold_dict, above_threshold_dict
