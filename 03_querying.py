import os
import json
import azure
from azure.search.documents import SearchClient
from rag_experiment_accelerator.config import Config
from rag_experiment_accelerator.llm.embeddings.base import EmbeddingsModel
from rag_experiment_accelerator.evaluation.search_eval import evaluate_search_result
from rag_experiment_accelerator.evaluation.spacy_evaluator import SpacyEvaluator
from rag_experiment_accelerator.utils.utils import get_index_name

from dotenv import load_dotenv


load_dotenv(override=True)

from rag_experiment_accelerator.ingest_data.acs_ingest import (
    we_need_multiple_questions,
    do_we_need_multiple_questions,
)
from rag_experiment_accelerator.search_type.acs_search_methods import (
    search_for_match_pure_vector_multi,
    search_for_match_semantic,
    search_for_match_Hybrid_multi,
    search_for_match_Hybrid_cross,
    search_for_match_text,
    search_for_match_pure_vector,
    search_for_match_pure_vector_cross,
    search_for_manual_hybrid,
)
from rag_experiment_accelerator.search_type.acs_search_methods import create_client
from rag_experiment_accelerator.llm.prompts import main_prompt_instruction
from rag_experiment_accelerator.llm.prompt_execution import generate_response
from rag_experiment_accelerator.data_assets.data_asset import create_data_asset
from rag_experiment_accelerator.reranking.reranker import (
    llm_rerank_documents,
    cross_encoder_rerank_documents,
)

from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)

search_mapping = {
    "search_for_match_semantic": search_for_match_semantic,
    "search_for_match_Hybrid_multi": search_for_match_Hybrid_multi,
    "search_for_match_Hybrid_cross": search_for_match_Hybrid_cross,
    "search_for_match_text": search_for_match_text,
    "search_for_match_pure_vector": search_for_match_pure_vector,
    "search_for_match_pure_vector_multi": search_for_match_pure_vector_multi,
    "search_for_match_pure_vector_cross": search_for_match_pure_vector_cross,
    "search_for_manual_hybrid": search_for_manual_hybrid,
}


def query_acs(
    search_client: azure.search.documents.SearchClient,
    user_prompt: str,
    s_v: str,
    retrieve_num_of_documents: str,
    embedding_model: EmbeddingsModel,
):
    """
    Queries the Azure Cognitive Search service using the specified search client and search parameters.

    Args:
        search_client (azure.search.documents.SearchClient): The Azure Cognitive Search client to use for querying the service.
        dimension (int): The dimension to search within.
        user_prompt (str): The user's search query.
        s_v (str): The version of the search service to use.
        retrieve_num_of_documents (int): The number of documents to retrieve.
        model_name (str): The name of the model to use for searching.

    Returns:
        list: A list of documents matching the search query.
    """
    if s_v not in search_mapping:
        pass

    return search_mapping[s_v](
        client=search_client,
        query=user_prompt,
        retrieve_num_of_documents=retrieve_num_of_documents,
        embedding_model=embedding_model
    )


def rerank_documents(
    docs: list[str],
    user_prompt: str,
    output_prompt: str,
    config: Config,
) -> list[str]:
    """
    Reranks a list of documents based on a given user prompt and configuration.

    Args:
        docs (list[str]): A list of documents to be reranked.
        user_prompt (str): The user prompt to be used for reranking.
        output_prompt (str): The output prompt to be used for reranking.
        config (Config): A configuration object containing reranking parameters.

    Returns:
        list[str]: A list of reranked documents.
    """
    result = []
    if config.RERANK_TYPE == "llm":
        result = llm_rerank_documents(
            docs,
            user_prompt,
            config.CHAT_MODEL_NAME,
            config.TEMPERATURE,
            config.LLM_RERANK_THRESHOLD,
        )
    elif config.RERANK_TYPE == "crossencoder":
        result = cross_encoder_rerank_documents(
            docs,
            user_prompt,
            output_prompt,
            config.CROSSENCODER_MODEL,
            config.CROSSENCODER_AT_K,
        )

    return result


def query_and_eval_acs(
    search_client: SearchClient,
    query: str,
    search_type: str,
    evaluation_content: str,
    retrieve_num_of_documents: int,
    evaluator: SpacyEvaluator,
    embedding_model: EmbeddingsModel,
) -> tuple[list[str], list[dict[str, any]]]:
    """
    Queries the Azure Cognitive Search service using the provided search client and parameters, and evaluates the search
    results using the provided evaluator and evaluation content. Returns a tuple containing the retrieved documents and
    the evaluation results.

    Args:
        search_client (SearchClient): The Azure Cognitive Search client to use for querying the service.
        dimension (int): The dimension of the search index to query.
        query (str): The search query to execute.
        search_type (str): The type of search to execute (e.g. 'semantic', 'vector', etc.).
        evaluation_content (str): The content to use for evaluating the search results.
        retrieve_num_of_documents (int): The number of documents to retrieve from the search results.
        evaluator (SpacyEvaluator): The evaluator to use for evaluating the search results.
        model_name (str): The name of the model to use for searching.

    Returns:
        tuple[list[dict[str, any]], dict[str, any]]: A tuple containing the retrieved documents and the evaluation results.
    """
    search_result = query_acs(
        search_client=search_client,
        user_prompt=query,
        s_v=search_type,
        retrieve_num_of_documents=retrieve_num_of_documents,
        embedding_model=embedding_model,
    )
    docs, evaluation = evaluate_search_result(
        search_result, evaluation_content, evaluator
    )
    evaluation["query"] = query
    return docs, evaluation


def query_and_eval_acs_multi(
    search_client: SearchClient,
    questions: list[str],
    original_prompt: str,
    output_prompt: str,
    search_type: str,
    evaluation_content: str,
    chat_model_name: str,
    temperature: float,
    evaluator: SpacyEvaluator,
    main_prompt_instruction: str,
    embedding_model: EmbeddingsModel,
) -> tuple[list[str], list[dict[str, any]]]:
    """
    Queries the Azure Cognitive Search service with multiple questions, evaluates the results, and generates a response
    using OpenAI's GPT-3 model.

    Args:
        search_client (SearchClient): The Azure Cognitive Search client.
        dimension (int): The number of dimensions in the embedding space.
        questions (list[str]): A list of questions to query the search service with.
        original_prompt (str): The original prompt to generate the response from.
        output_prompt (str): The output prompt to use for reranking the search results.
        search_type (str): The type of search to perform (e.g. 'semantic', 'exact').
        evaluation_content (str): The content to use for evaluation.
        config (Config): The configuration object.
        evaluator (SpacyEvaluator): The evaluator object.

    Returns:
        tuple[list[str], list[dict[str, any]]]: A tuple containing a list of OpenAI responses and a list of evaluation
        results for each question.
    """
    context = []
    evals = []
    for question in questions:
        docs, evaluation = query_and_eval_acs(
            search_client=search_client,
            query=question,
            search_type=search_type,
            evaluation_content=evaluation_content,
            retrieve_num_of_documents=config.RETRIEVE_NUM_OF_DOCUMENTS,
            evaluator=evaluator,
            embedding_model=embedding_model,
        )
        evals.append(evaluation)

        if config.RERANK:
            prompt_instruction_context = rerank_documents(
                docs, question, output_prompt, config
            )
        else:
            prompt_instruction_context = docs

        full_prompt_instruction = (
            main_prompt_instruction + "\n" + "\n".join(prompt_instruction_context)
        )
        full_prompt_instruction = (
            main_prompt_instruction + "\n" + "\n".join(prompt_instruction_context)
        )
        openai_response = generate_response(
            full_prompt_instruction,
            original_prompt,
            chat_model_name,
            temperature,
        )
        context.append(openai_response)
        logger.debug(openai_response)

    return context, evals


def main(config: Config):
    """
    Runs the main experiment loop, which evaluates a set of search configurations against a given dataset.

    Args:
        config (Config): A configuration object containing experiment parameters.

    Returns:
        None
    """
    service_endpoint = config.AzureSearchCredentials.AZURE_SEARCH_SERVICE_ENDPOINT
    search_admin_key = config.AzureSearchCredentials.AZURE_SEARCH_ADMIN_KEY
    jsonl_file_path = config.EVAL_DATA_JSON_FILE_PATH
    question_count = 0
    try:
        with open(jsonl_file_path, "r") as file:
            for line in file:
                question_count += 1

        if config.MAIN_PROMPT_INSTRUCTIONS:
            prompt_instruction = config.MAIN_PROMPT_INSTRUCTIONS
        else:
            prompt_instruction = main_prompt_instruction

        directory_path = "artifacts/outputs"
        os.makedirs(directory_path, exist_ok=True)

        evaluator = SpacyEvaluator(config.SEARCH_RELEVANCY_THRESHOLD)

        for chunk_size in config.CHUNK_SIZES:
            for overlap in config.OVERLAP_SIZES:
                for embedding_model in config.embedding_models:
                    for ef_construction in config.EF_CONSTRUCTIONS:
                        for ef_search in config.EF_SEARCHES:
                            index_name = get_index_name(
                                prefix=config.NAME_PREFIX,
                                chunk_size=chunk_size,
                                overlap=overlap,
                                embedding_model_name=embedding_model.model_name,
                                ef_construction=ef_construction,
                                ef_search=ef_search,
                            )
                            logger.info(f"Index: {index_name}")

                            write_path = (
                                f"{directory_path}/eval_output_{index_name}.jsonl"
                            )
                            if os.path.exists(write_path):
                                continue

                            search_client = create_client(
                                service_endpoint, index_name, search_admin_key
                            )

                            with open(jsonl_file_path, "r") as file:
                                for line in file:
                                    data = json.loads(line)
                                    user_prompt = data.get("user_prompt")
                                    output_prompt = data.get("output_prompt")
                                    qna_context = data.get("context", "")

                                    is_multi_question = do_we_need_multiple_questions(
                                        user_prompt,
                                        config.CHAT_MODEL_NAME,
                                        config.TEMPERATURE,
                                    )
                                    if is_multi_question:
                                        responses = json.loads(
                                            we_need_multiple_questions(
                                                user_prompt,
                                                config.CHAT_MODEL_NAME,
                                                config.TEMPERATURE,
                                            )
                                        )
                                        new_questions = []
                                        if isinstance(responses, dict):
                                            new_questions = responses["questions"]
                                        else:
                                            for response in responses:
                                                if "question" in response:
                                                    new_questions.append(
                                                        response["question"]
                                                    )
                                        new_questions.append(user_prompt)

                                    evaluation_content = user_prompt + qna_context
                                    for s_v in config.SEARCH_VARIANTS:
                                        search_evals = []
                                        if is_multi_question:
                                            (
                                                docs,
                                                search_evals,
                                            ) = query_and_eval_acs_multi(
                                                search_client=search_client,
                                                questions=new_questions,
                                                original_prompt=user_prompt,
                                                output_prompt=output_prompt,
                                                search_type=s_v,
                                                evaluation_content=evaluation_content,
                                                chat_model_name=config.CHAT_MODEL_NAME,
                                                temperature=config.TEMPERATURE,
                                                evaluator=evaluator,
                                                main_prompt_instruction=prompt_instruction,
                                                embedding_model=embedding_model
                                            )
                                        else:
                                            docs, evaluation = query_and_eval_acs(
                                                search_client=search_client,
                                                query=user_prompt,
                                                search_type=s_v,
                                                evaluation_content=evaluation_content,
                                                retrieve_num_of_documents=config.RETRIEVE_NUM_OF_DOCUMENTS,
                                                evaluator=evaluator,
                                                embedding_model=embedding_model
                                            )
                                            search_evals.append(evaluation)

                                        if config.RERANK:
                                            prompt_instruction_context = (
                                                rerank_documents(
                                                    docs,
                                                    user_prompt,
                                                    output_prompt,
                                                    config,
                                                )
                                            )
                                        else:
                                            prompt_instruction_context = docs

                                        full_prompt_instruction = (
                                            prompt_instruction
                                            + "\n"
                                            + "\n".join(prompt_instruction_context)
                                        )
                                        openai_response = generate_response(
                                            full_prompt_instruction,
                                            user_prompt,
                                            config.CHAT_MODEL_NAME,
                                            config.TEMPERATURE,
                                        )
                                        logger.debug(openai_response)

                                        output = {
                                            "rerank": config.RERANK,
                                            "rerank_type": config.RERANK_TYPE,
                                            "crossencoder_model": config.CROSSENCODER_MODEL,
                                            "llm_re_rank_threshold": config.LLM_RERANK_THRESHOLD,
                                            "retrieve_num_of_documents": config.RETRIEVE_NUM_OF_DOCUMENTS,
                                            "cross_encoder_at_k": config.CROSSENCODER_AT_K,
                                            "question_count": question_count,
                                            "actual": openai_response,
                                            "expected": output_prompt,
                                            "search_type": s_v,
                                            "search_evals": search_evals,
                                        }

                                        with open(write_path, "a") as out:
                                            json_string = json.dumps(output)
                                            out.write(json_string + "\n")

                            search_client.close()
                            create_data_asset(
                                write_path, index_name, config.AzureMLCredentials
                            )
    except FileNotFoundError:
        logger.error("The file does not exist: " + jsonl_file_path)


if __name__ == "__main__":
    config = Config()
    main(config)
