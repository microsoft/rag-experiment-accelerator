import os
import json
import azure
from azure.search.documents import SearchClient
from openai import BadRequestError
from rag_experiment_accelerator.config import Config
from rag_experiment_accelerator.evaluation.search_eval import evaluate_search_result
from rag_experiment_accelerator.evaluation.spacy_evaluator import SpacyEvaluator
from rag_experiment_accelerator.run.args import get_directory_arg
from rag_experiment_accelerator.utils.auth import get_default_az_cred

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
    dimension: int,
    user_prompt: str,
    s_v: str,
    retrieve_num_of_documents: str,
    model_name: str = None,
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
        size=dimension,
        query=user_prompt,
        retrieve_num_of_documents=retrieve_num_of_documents,
        model_name=model_name,
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
            config.AOAI_DEPLOYMENT_NAME,
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
    dimension: int,
    query: str,
    search_type: str,
    evaluation_content: str,
    retrieve_num_of_documents: int,
    evaluator: SpacyEvaluator,
    model_name: str = None,
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
        dimension=dimension,
        user_prompt=query,
        s_v=search_type,
        retrieve_num_of_documents=retrieve_num_of_documents,
        model_name=model_name,
    )
    docs, evaluation = evaluate_search_result(
        search_result, evaluation_content, evaluator
    )
    evaluation["query"] = query
    return docs, evaluation


def query_and_eval_acs_multi(
    search_client: SearchClient,
    dimension: int,
    questions: list[str],
    original_prompt: str,
    output_prompt: str,
    search_type: str,
    evaluation_content: str,
    config: Config,
    evaluator: SpacyEvaluator,
    main_prompt_instruction: str,
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
        main_prompt_instruction (str): The main prompt instruction for the query

    Returns:
        tuple[list[str], list[dict[str, any]]]: A tuple containing a list of OpenAI responses and a list of evaluation
        results for each question.
    """
    context = []
    evals = []
    for question in questions:
        docs, evaluation = query_and_eval_acs(
            search_client=search_client,
            dimension=dimension,
            query=question,
            search_type=search_type,
            evaluation_content=evaluation_content,
            retrieve_num_of_documents=config.RETRIEVE_NUM_OF_DOCUMENTS,
            evaluator=evaluator,
            model_name=config.EMBEDDING_MODEL_NAME,
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
        openai_response = generate_response(
            full_prompt_instruction,
            original_prompt,
            config.AOAI_DEPLOYMENT_NAME,
            config.TEMPERATURE,
        )
        context.append(openai_response)
        logger.debug(openai_response)

    return context, evals


def run(config_dir: str):
    """
    Runs the main experiment loop, which evaluates a set of search configurations against a given dataset.

    Returns:
        None
    """
    config = Config(config_dir)
    service_endpoint = config.AzureSearchCredentials.AZURE_SEARCH_SERVICE_ENDPOINT
    search_admin_key = config.AzureSearchCredentials.AZURE_SEARCH_ADMIN_KEY
    question_count = 0
    # ensure we have a valid Azure credential before going throught the loop.
    azure_cred = get_default_az_cred()
    try:
        with open(config.EVAL_DATA_JSONL_FILE_PATH, "r") as file:
            for line in file:
                question_count += 1

        try:
            output_dir = f"{config.artifacts_dir}/outputs"
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Unable to create the '{output_dir}' directory. Please ensure you have the proper permissions and try again")
            raise e

        evaluator = SpacyEvaluator(config.SEARCH_RELEVANCY_THRESHOLD)

        for config_item in config.CHUNK_SIZES:
            for overlap in config.OVERLAP_SIZES:
                for dimension in config.EMBEDDING_DIMENSIONS:
                    for ef_construction in config.EF_CONSTRUCTIONS:
                        for ef_search in config.EF_SEARCHES:
                            index_name = f"{config.NAME_PREFIX}-{config_item}-{overlap}-{dimension}-{ef_construction}-{ef_search}"
                            logger.info(f"Index: {index_name}")

                            write_path = (
                                f"{output_dir}/eval_output_{index_name}.jsonl"
                            )
                            if os.path.exists(write_path):
                                continue

                            search_client = create_client(
                                service_endpoint, index_name, search_admin_key
                            )

                            with open(config.EVAL_DATA_JSONL_FILE_PATH, "r") as file:
                                for line in file:
                                    data = json.loads(line)
                                    user_prompt = data.get("user_prompt")
                                    output_prompt = data.get("output_prompt")
                                    qna_context = data.get("context", "")

                                    is_multi_question = do_we_need_multiple_questions(
                                        user_prompt,
                                        config.AOAI_DEPLOYMENT_NAME,
                                        config.TEMPERATURE,
                                    )
                                    if is_multi_question:
                                        responses = json.loads(
                                            we_need_multiple_questions(
                                                user_prompt,
                                                config.AOAI_DEPLOYMENT_NAME,
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
                                    try:
                                        for s_v in config.SEARCH_VARIANTS:
                                            search_evals = []
                                            if is_multi_question:
                                                (
                                                    docs,
                                                    search_evals,
                                                ) = query_and_eval_acs_multi(
                                                    search_client,
                                                    dimension,
                                                    new_questions,
                                                    user_prompt,
                                                    output_prompt,
                                                    s_v,
                                                    evaluation_content,
                                                    config,
                                                    evaluator,
                                                    config.MAIN_PROMPT_INSTRUCTION,
                                                )
                                            else:
                                                docs, evaluation = query_and_eval_acs(
                                                    search_client=search_client,
                                                    dimension=dimension,
                                                    query=user_prompt,
                                                    search_type=s_v,
                                                    evaluation_content=evaluation_content,
                                                    retrieve_num_of_documents=config.RETRIEVE_NUM_OF_DOCUMENTS,
                                                    evaluator=evaluator,
                                                    model_name=config.EMBEDDING_MODEL_NAME
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
                                                config.MAIN_PROMPT_INSTRUCTION
                                                + "\n"
                                                + "\n".join(prompt_instruction_context)
                                            )
                                            openai_response = generate_response(
                                                full_prompt_instruction,
                                                user_prompt,
                                                config.AOAI_DEPLOYMENT_NAME,
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
                                                "context": qna_context,
                                            }

                                            with open(write_path, "a") as out:
                                                json_string = json.dumps(output)
                                                out.write(json_string + "\n")
                                    except BadRequestError as e:
                                        logger.error(
                                            f"Invalid request. Skipping question: {user_prompt}",
                                            exc_info=e,
                                        )
                                        continue

                            search_client.close()
                            create_data_asset(
                                write_path, index_name, azure_cred, config.AzureMLCredentials
                            )
    except FileNotFoundError:
        logger.error("The file does not exist: " + config.EVAL_DATA_JSONL_FILE_PATH)
