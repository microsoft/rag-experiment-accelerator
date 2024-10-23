import hashlib


def score_to_similarity(score):
    if score == 0:
        return 0
    cosineDistance = (1 - score) / score
    return -cosineDistance + 1


def getHashDocId( prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()


def getResult(results, threshold):
    highest_score = float('-inf')
    similar_result = None

    for result in results:
        score = result["@search.score"]
        # print("Original score is " + str(score))
        similarity_score = score_to_similarity(score)
        # print("Transformed similarity score is " + str(similarity_score))

        if similarity_score >= threshold:
            context_item = {
                "@search.score": score,
                "content": result["content"],
                "prompt_text": result["prompt_text"],
                "similarity_score": similarity_score
            }

            if similarity_score > highest_score:
                highest_score = similarity_score
                similar_result = context_item

    return similar_result


def get_doc_id_from_result(results):
    doc_ids = []
    try:
        for result in results:
            doc_id = result["id"]
            if doc_id:
                doc_ids.append(doc_id)  # Add valid doc IDs to the list
    except Exception as e:
        print(f"Error while processing results: {str(e)}")
    return doc_ids