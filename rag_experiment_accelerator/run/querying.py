from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import json

from azure.search.documents import SearchClient
from dotenv import load_dotenv
from openai import BadRequestError

from rag_experiment_accelerator.artifact.handlers.query_output_handler import (
    QueryOutputHandler,
)
from rag_experiment_accelerator.artifact.models.query_output import QueryOutput
from rag_experiment_accelerator.config.config import Config
from rag_experiment_accelerator.config.index_config import IndexConfig
from rag_experiment_accelerator.evaluation.eval import cosine_similarity
from rag_experiment_accelerator.embedding.embedding_model import EmbeddingModel
from rag_experiment_accelerator.evaluation.search_eval import (
    evaluate_search_result,
)
from rag_experiment_accelerator.evaluation.spacy_evaluator import (
    SpacyEvaluator,
)

from rag_experiment_accelerator.ingest_data.acs_ingest import (
    do_we_need_multiple_questions,
    we_need_multiple_questions,
)
from rag_experiment_accelerator.llm.exceptions import ContentFilteredException
from rag_experiment_accelerator.llm.response_generator import ResponseGenerator
from rag_experiment_accelerator.reranking.reranker import (
    cross_encoder_rerank_documents,
    llm_rerank_documents,
)
from rag_experiment_accelerator.search_type.acs_search_methods import (
    create_client,
    search_for_manual_hybrid,
    search_for_match_Hybrid_cross,
    search_for_match_Hybrid_multi,
    search_for_match_pure_vector,
    search_for_match_pure_vector_cross,
    search_for_match_pure_vector_multi,
    search_for_match_semantic,
    search_for_match_text,
)
from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.llm.prompts import (
    prompt_generated_hypothetical_answer,
    prompt_generated_hypothetical_document_to_answer,
    prompt_generated_related_questions,
)

load_dotenv(override=True)


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
    search_client: SearchClient,
    embedding_model: EmbeddingModel,
    user_prompt: str,
    s_v: str,
    retrieve_num_of_documents: str,
):
    """
    Queries the Azure AI Search service using the specified search client and search parameters.

    Args:
        search_client (SearchClient): The Azure AI Search client to use for querying the service.
        embedding_model (EmbeddingModel): The model used to generate the embeddings.
        user_prompt (str): The user's search query.
        s_v (str): The version of the search service to use.
        retrieve_num_of_documents (int): The number of documents to retrieve.

    Returns:
        list: A list of documents matching the search query.
    """
    if s_v not in search_mapping:
        pass

    return search_mapping[s_v](
        client=search_client,
        embedding_model=embedding_model,
        query=user_prompt,
        retrieve_num_of_documents=retrieve_num_of_documents,
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
            config.AZURE_OAI_CHAT_DEPLOYMENT_NAME,
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


def hyde(
    config: Config,
    response_generator: ResponseGenerator,
    queries: list[str],
):
    if config.HYDE == "disabled":
        return queries

    generated_queries = []
    for query in queries:
        if config.HYDE == "generated_hypothetical_answer":
            result = response_generator.generate_response(
                prompt_generated_hypothetical_answer,
                query,
            )
        elif config.HYDE == "generated_hypothetical_document_to_answer":
            result = response_generator.generate_response(
                prompt_generated_hypothetical_document_to_answer,
                query,
            )
        else:
            raise NotImplementedError(
                f"configuration for hyde with value of [{config.HYDE}] is not supported"
            )
        generated_queries.append(result)
    return generated_queries


def query_expansion(
    config: Config,
    response_generator: ResponseGenerator,
    embedding_model: EmbeddingModel,
    query: str,
) -> list[str]:
    # Query expansion with generated questions
    augmented_questions = response_generator.generate_response(
        prompt_generated_related_questions,
        query,
    )

    # Filter out non related questions
    questions = filter_non_related_questions(
        query,
        augmented_questions.split("\n"),
        embedding_model,
        config.MIN_QUERY_EXPANSION_RELATED_QUESTION_SIMILARITY_SCORE,
    )

    return questions


def dedupulicate_search_results(search_results: list[dict]) -> list[dict]:
    doc_set = set()
    score_dict = {}

    # deduplicate and sort retrieved documents by using a set
    for doc in search_results:
        doc_set.add(doc["content"])
        score_dict[doc["content"]] = doc["@search.score"]

    search_result = list(doc_set)
    search_result = [
        {"content": doc, "@search.score": score_dict[doc]} for doc in search_result
    ]
    search_result.sort(key=lambda x: x["@search.score"], reverse=True)

    return search_result


def query_and_eval_acs(
    search_client: SearchClient,
    embedding_model: EmbeddingModel,
    query: str,
    search_type: str,
    evaluation_content: str,
    retrieve_num_of_documents: int,
    evaluator: SpacyEvaluator,
    config: Config,
    response_generator: ResponseGenerator,
) -> tuple[list[str], list[dict[str, any]]]:
    """
    Queries the Azure AI Search service using the provided search client and parameters, and evaluates the search
    results using the provided evaluator and evaluation content. Returns a tuple containing the retrieved documents and
    the evaluation results.

    Args:
        search_client (SearchClient): The Azure AI Search client to use for querying the service.
        embedding_model (EmbeddingModel): The model used to generate the embeddings.
        query (str): The search query to execute.
        search_type (str): The type of search to execute (e.g. 'semantic', 'vector', etc.).
        evaluation_content (str): The content to use for evaluating the search results.
        retrieve_num_of_documents (int): The number of documents to retrieve from the search results.
        evaluator (SpacyEvaluator): The evaluator to use for evaluating the search results.
        config (Config): The configuration object.
        response_generator (ResponseGenerator): The response generator object.

    Returns:
        tuple[list[dict[str, any]], dict[str, any]]: A tuple containing the retrieved documents and the evaluation results.
    """

    if config.QUERY_EXPANSION:
        generated_queries = query_expansion(
            config, response_generator, embedding_model, query
        )
    else:
        generated_queries = [query]

    generated_queries = hyde(config, response_generator, generated_queries)
    search_results = []
    for generated_query in generated_queries:
        search_result = query_acs(
            search_client=search_client,
            embedding_model=embedding_model,
            user_prompt=generated_query,
            s_v=search_type,
            retrieve_num_of_documents=retrieve_num_of_documents,
        )
        search_results.extend(search_result)

    search_results = dedupulicate_search_results(search_results)
    search_result = search_result[: config.RETRIEVE_NUM_OF_DOCUMENTS]

    docs, evaluation = evaluate_search_result(
        search_results, evaluation_content, evaluator
    )

    evaluation["query"] = query
    return docs, evaluation


def filter_non_related_questions(
    query,
    generated_questions,
    embedding_model,
    MIN_QUERY_EXPANSION_RELATED_QUESTION_SIMILARITY_SCORE,
):
    questions = [query]

    query_vector = embedding_model.generate_embedding(query)

    for generated_question in generated_questions:
        generated_question_vector = embedding_model.generate_embedding(
            generated_question
        )
        similarity_score_array = (
            cosine_similarity(query_vector, generated_question_vector) * 100
        )
        similarity_score = int(
            sum(similarity_score_array) / len(similarity_score_array)
        )
        if similarity_score >= MIN_QUERY_EXPANSION_RELATED_QUESTION_SIMILARITY_SCORE:
            questions.append(generated_question)

    return questions


def query_and_eval_acs_multi(
    search_client: SearchClient,
    embedding_model: EmbeddingModel,
    questions: list[str],
    original_prompt: str,
    output_prompt: str,
    search_type: str,
    evaluation_content: str,
    environment: Environment,
    config: Config,
    evaluator: SpacyEvaluator,
    main_prompt_instruction: str,
) -> tuple[list[str], list[dict[str, any]]]:
    """
    Queries the Azure AI Search service with multiple questions, evaluates the results, and generates a response
    using OpenAI's GPT-3 model.

    Args:
        search_client (SearchClient): The Azure AI Search client.
        embedding_model (EmbeddingModel): The model used to generate the embeddings.
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
    evaluations = []
    response_generator = ResponseGenerator(
        environment, config, config.AZURE_OAI_CHAT_DEPLOYMENT_NAME
    )
    for question in questions:
        docs, evaluation = query_and_eval_acs(
            search_client=search_client,
            embedding_model=embedding_model,
            query=question,
            search_type=search_type,
            evaluation_content=evaluation_content,
            retrieve_num_of_documents=config.RETRIEVE_NUM_OF_DOCUMENTS,
            evaluator=evaluator,
            config=config,
            response_generator=response_generator,
        )
        if len(docs) == 0:
            logger.warning(f"No documents found for question: {question}")
            continue

        evaluations.append(evaluation)

        if config.RERANK:
            prompt_instruction_context = rerank_documents(
                docs, question, output_prompt, config
            )
        else:
            prompt_instruction_context = docs

        full_prompt_instruction = (
            main_prompt_instruction + "\n" + "\n".join(prompt_instruction_context)
        )
        openai_response = response_generator.generate_response(
            full_prompt_instruction,
            original_prompt,
        )
        context.append(openai_response)
        logger.debug(openai_response)

    return context, evaluations


def query_and_eval_single_line(
    line: str,
    line_number: int,
    handler: QueryOutputHandler,
    environment: Environment,
    config: Config,
    index_config: IndexConfig,
    response_generator: ResponseGenerator,
    search_client: SearchClient,
    evaluator: SpacyEvaluator,
    question_count: int,
):
    logger.info(f"Processing question {line_number + 1} out of {question_count}\n\n")
    data = json.loads(line)
    user_prompt = data.get("user_prompt")
    output_prompt = data.get("output_prompt")
    qna_context = data.get("context", "")

    is_multi_question = (
        config.EXPAND_TO_MULTIPLE_QUESTIONS
        and do_we_need_multiple_questions(user_prompt, response_generator, config)
    )

    if is_multi_question:
        try:
            llm_response = we_need_multiple_questions(user_prompt, response_generator)
            responses = json.loads(llm_response)
            new_questions = []
            if isinstance(responses, dict):
                new_questions = responses["questions"]
            else:
                for response in responses:
                    if "question" in response:
                        new_questions.append(response["question"])
            new_questions.append(user_prompt)
        except ContentFilteredException as e:
            logger.error(
                f"Content Filtered. Unable to generate multiple questions for: {user_prompt}",
                exc_info=e,
            )
            is_multi_question = False

    evaluation_content = user_prompt + qna_context

    try:
        for s_v in config.SEARCH_VARIANTS:
            search_evals = []
            if is_multi_question:
                (
                    docs,
                    search_evals,
                ) = query_and_eval_acs_multi(
                    search_client=search_client,
                    embedding_model=index_config.embedding_model,
                    questions=new_questions,
                    original_prompt=user_prompt,
                    output_prompt=output_prompt,
                    search_type=s_v,
                    evaluation_content=evaluation_content,
                    environment=environment,
                    config=config,
                    evaluator=evaluator,
                    main_prompt_instruction=config.MAIN_PROMPT_INSTRUCTION,
                )
            else:
                (
                    docs,
                    evaluation,
                ) = query_and_eval_acs(
                    search_client=search_client,
                    embedding_model=index_config.embedding_model,
                    query=user_prompt,
                    search_type=s_v,
                    evaluation_content=evaluation_content,
                    retrieve_num_of_documents=config.RETRIEVE_NUM_OF_DOCUMENTS,
                    evaluator=evaluator,
                    config=config,
                    response_generator=response_generator,
                )
                search_evals.append(evaluation)
            if config.RERANK and len(docs) > 0:
                prompt_instruction_context = rerank_documents(
                    docs,
                    user_prompt,
                    output_prompt,
                    config,
                )
            else:
                prompt_instruction_context = docs

            full_prompt_instruction = (
                config.MAIN_PROMPT_INSTRUCTION
                + "\n"
                + "\n".join(prompt_instruction_context)
            )
            openai_response = response_generator.generate_response(
                full_prompt_instruction,
                user_prompt,
            )
            logger.debug(openai_response)

            output = QueryOutput(
                rerank=config.RERANK,
                rerank_type=config.RERANK_TYPE,
                crossencoder_model=config.CROSSENCODER_MODEL,
                llm_re_rank_threshold=config.LLM_RERANK_THRESHOLD,
                retrieve_num_of_documents=config.RETRIEVE_NUM_OF_DOCUMENTS,
                crossencoder_at_k=config.CROSSENCODER_AT_K,
                question_count=question_count,
                actual=openai_response,
                expected=output_prompt,
                search_type=s_v,
                search_evals=search_evals,
                context=qna_context,
                question=user_prompt,
            )
            handler.save(
                index_name=index_config.index_name(),
                data=output,
                experiment_name=config.EXPERIMENT_NAME,
                job_name=config.JOB_NAME,
            )
    except BadRequestError as e:
        logger.error(
            "Invalid request. Skipping question: {user_prompt}",
            exc_info=e,
        )


def run(environment: Environment, config: Config, index_config: IndexConfig):
    """
    Runs the main experiment loop, which evaluates a set of search configurations against a given dataset.

    Returns:
        None
    """
    question_count = 0
    try:
        with open(config.EVAL_DATA_JSONL_FILE_PATH, "r") as file:
            for line in file:
                question_count += 1
    except FileNotFoundError as e:
        logger.error("The file does not exist: " + config.EVAL_DATA_JSONL_FILE_PATH)
        raise e

    evaluator = SpacyEvaluator(config.SEARCH_RELEVANCY_THRESHOLD)
    handler = QueryOutputHandler(config.QUERY_DATA_LOCATION)
    response_generator = ResponseGenerator(
        environment, config, config.AZURE_OAI_CHAT_DEPLOYMENT_NAME
    )
    for index_config in config.index_configs():
        logger.info(f"Processing index: {index_config.index_name()}")

        handler.handle_archive_by_index(
            index_config.index_name(), config.EXPERIMENT_NAME, config.JOB_NAME
        )

        search_client = create_client(
            environment.azure_search_service_endpoint,
            index_config.index_name(),
            environment.azure_search_admin_key,
        )
        with open(config.EVAL_DATA_JSONL_FILE_PATH, "r") as file:
            with ExitStack() as stack:
                executor = stack.enter_context(
                    ThreadPoolExecutor(config.MAX_WORKER_THREADS)
                )
                futures = {
                    executor.submit(
                        query_and_eval_single_line,
                        line,
                        line_number,
                        handler,
                        environment,
                        config,
                        index_config,
                        response_generator,
                        search_client,
                        evaluator,
                        question_count,
                    ): line
                    for line_number, line in enumerate(file)
                }

                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        logger.error(
                            f"query generated an exception: {exc} for line {line}..."
                        )

        search_client.close()
