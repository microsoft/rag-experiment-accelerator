import re

from sentence_transformers import CrossEncoder

from rag_experiment_accelerator.llm.prompt import rerank_prompt_instruction
from rag_experiment_accelerator.llm.response_generator import ResponseGenerator
from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


def cross_encoder_rerank_documents(
    documents, user_prompt, output_prompt, model_name, k
):
    """
    Reranks a list of documents based on their relevance to a user prompt using a cross-encoder model.

    Args:
        documents (list): A list of documents to be reranked.
        user_prompt (str): The user prompt to be used as the query.
        output_prompt (str): The output prompt to be used as the context.
        model_name (str): The name of the pre-trained cross-encoder model to be used.
        k (int): The number of top documents to be returned.

    Returns:
        list: A list of the top k documents, sorted by their relevance to the user prompt.
    """
    if not documents:
        return []

    model = CrossEncoder(model_name)
    cross_scores_ques = model.predict(
        [[user_prompt, item] for item in documents],
        apply_softmax=True,
        convert_to_numpy=True,
    )

    top_indices_ques = cross_scores_ques.argsort()[-k:][::-1]
    sub_context = []
    for idx in list(top_indices_ques):
        sub_context.append(documents[idx])

    return sub_context


def llm_rerank_documents(
    documents, question, response_generator: ResponseGenerator, rerank_threshold
):
    """
    Reranks a list of documents based on a given question using the LLM model.

    Args:
        documents (list): A list of documents to be reranked.
        question (str): The question to be used for reranking.
        response_generator (ResponseGenerator): The initialised ResponseGenerator to use.
        rerank_threshold (int): The threshold for reranking documents.

    Returns:
        list: A list of reranked documents.
    """
    rerank_context = ""
    for index, docs in enumerate(documents):
        rerank_context += "\ndocument " + str(index) + ":\n"
        rerank_context += docs + "\n"

    response: dict[str, int] | None = response_generator.generate_response(
        rerank_prompt_instruction,
        documents=rerank_context,
        question=question,
    )

    logger.debug("Reranker response:\n", response)

    if response is None:
        return documents

    result = []
    for key, _ in sorted(response.items(), key=lambda x: x[1], reverse=True):
        document_index = int(re.search(r"document_(\d+)", key))
        result.append(documents[document_index])

    return result
