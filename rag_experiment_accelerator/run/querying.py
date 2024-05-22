from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import json
import numpy as np
from azure.search.documents import SearchClient
from dotenv import load_dotenv
import mlflow
from openai import BadRequestError

from rag_experiment_accelerator.artifact.handlers.query_output_handler import (
    QueryOutputHandler,
)
from rag_experiment_accelerator.artifact.models.query_output import QueryOutput
from rag_experiment_accelerator.checkpoint import cache_with_checkpoint
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

from rag_experiment_accelerator.llm.response_generator import ResponseGenerator
from rag_experiment_accelerator.llm.prompt import (
    prompt_generate_hypothetical_answer,
    prompt_generate_hypothetical_document,
    prompt_generate_hypothetical_questions,
    main_instruction,
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
    response_generator: ResponseGenerator,
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
    if config.rerank_type == "llm":
        result = llm_rerank_documents(
            docs,
            user_prompt,
            response_generator,
            config.llm_rerank_threshold,
        )
    elif config.rerank_type == "crossencoder":
        result = cross_encoder_rerank_documents(
            docs,
            user_prompt,
            output_prompt,
            config.crossencoder_model,
            config.crossencoder_at_k,
        )

    return result


def hyde(
    config: Config,
    response_generator: ResponseGenerator,
    queries: list[str],
):
    if config.hyde == "disabled":
        return queries

    generated_queries = []
    for query in queries:
        if config.hyde == "generated_hypothetical_answer":
            result = response_generator.generate_response(
                prompt_generate_hypothetical_answer,
                text=query,
            )
        elif config.hyde == "generated_hypothetical_document_to_answer":
            result = response_generator.generate_response(
                prompt_generate_hypothetical_document,
                text=query,
            )
        else:
            raise NotImplementedError(
                f"configuration for hyde with value of [{config.hyde}] is not supported"
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
        prompt_generate_hypothetical_questions,
        text=query,
        prompt_last=True,
    )

    if not augmented_questions:
        return [query]

    # Filter out non related questions
    questions = filter_non_related_questions(
        query,
        augmented_questions,
        embedding_model,
        config.min_query_expansion_related_question_similarity_score,
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

    if config.query_expansion:
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
    search_result = search_result[: config.retrieve_num_of_documents]

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
            cosine_similarity(
                np.array(query_vector).reshape(1, -1),
                np.array(generated_question_vector).reshape(1, -1),
            )
            * 100
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
    config: Config,
    evaluator: SpacyEvaluator,
    response_generator: ResponseGenerator,
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

    Returns:
        tuple[list[str], list[dict[str, any]]]: A tuple containing a list of OpenAI responses and a list of evaluation
        results for each question.
    """
    context = []
    evaluations = []

    for question in questions:
        docs, evaluation = query_and_eval_acs(
            search_client=search_client,
            embedding_model=embedding_model,
            query=question,
            search_type=search_type,
            evaluation_content=evaluation_content,
            retrieve_num_of_documents=config.retrieve_num_of_documents,
            evaluator=evaluator,
            config=config,
            response_generator=response_generator,
        )
        if len(docs) == 0:
            logger.warning(f"No documents found for question: {question}")
            continue

        evaluations.append(evaluation)

        if config.rerank:
            prompt_instruction_context = rerank_documents(
                docs, question, output_prompt, config, response_generator
            )
        else:
            prompt_instruction_context = docs

        # TODO: Here was a bug, caused by the fact that we are not limiting the number of documents to retrieve
        # Current solution is just forcefully limiting the number of documents to retrieve assuming they are sorted
        if len(prompt_instruction_context) > config.retrieve_num_of_documents:
            prompt_instruction_context = prompt_instruction_context[
                : config.retrieve_num_of_documents
            ]

        request_context = "\n".join(prompt_instruction_context)
        request_question = original_prompt

        openai_response = response_generator.generate_response(
            main_instruction,
            context=request_context,
            question=request_question,
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
    data: dict[str, any] = json.loads(line)
    user_prompt = data.get("user_prompt")
    output_prompt = data.get("output_prompt")
    qna_context = data.get("context", "")

    is_multi_question = (
        config.expand_to_multiple_questions
        and do_we_need_multiple_questions(user_prompt, response_generator, config)
    )

    new_questions = []
    if is_multi_question:
        new_questions = we_need_multiple_questions(user_prompt, response_generator)

        if new_questions is None:
            logger.warning(
                f"Unable to generate multiple questions for: {user_prompt}. Skipping..."
            )
            is_multi_question = False
        else:
            new_questions.append(user_prompt)

    evaluation_content = user_prompt + qna_context

    try:
        for s_v in config.search_types:
            output = get_query_output(
                environment,
                config,
                index_config,
                response_generator,
                search_client,
                evaluator,
                question_count,
                user_prompt,
                output_prompt,
                qna_context,
                is_multi_question,
                new_questions,
                evaluation_content,
                s_v,
            )
            handler.save(
                index_name=index_config.index_name(),
                data=output,
                experiment_name=config.experiment_name,
                job_name=config.job_name,
            )
    except BadRequestError as e:
        logger.error(
            "Invalid request. Skipping question: {user_prompt}",
            exc_info=e,
        )


@cache_with_checkpoint(
    key="user_prompt+output_prompt+qna_context+index_config.index_name()"
)
def get_query_output(
    environment,
    config,
    index_config,
    response_generator,
    search_client,
    evaluator,
    question_count,
    user_prompt,
    output_prompt,
    qna_context,
    is_multi_question,
    new_questions,
    evaluation_content,
    s_v,
):
    search_evals = []

    response_generator = ResponseGenerator(
        environment, config, config.azure_oai_chat_deployment_name
    )

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
            config=config,
            evaluator=evaluator,
            response_generator=response_generator,
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
            retrieve_num_of_documents=config.retrieve_num_of_documents,
            evaluator=evaluator,
            config=config,
            response_generator=response_generator,
        )
        search_evals.append(evaluation)
    if config.rerank and len(docs) > 0:
        prompt_instruction_context = rerank_documents(
            docs,
            user_prompt,
            output_prompt,
            config,
            response_generator,
        )
    else:
        prompt_instruction_context = docs

    openai_response = response_generator.generate_response(
        main_instruction,
        context="\n".join(prompt_instruction_context),
        question=user_prompt,
    )

    output = QueryOutput(
        rerank=config.rerank,
        rerank_type=config.rerank_type,
        crossencoder_model=config.crossencoder_model,
        llm_re_rank_threshold=config.llm_rerank_threshold,
        retrieve_num_of_documents=config.retrieve_num_of_documents,
        crossencoder_at_k=config.crossencoder_at_k,
        question_count=question_count,
        actual=openai_response,
        expected=output_prompt,
        search_type=s_v,
        search_evals=search_evals,
        context=qna_context,
        question=user_prompt,
    )

    return output


def run(
    environment: Environment,
    config: Config,
    index_config: IndexConfig,
    mlflow_client: mlflow.MlflowClient,
):
    """
    Runs the main experiment loop, which evaluates a set of search configurations against a given dataset.

    Returns:
        None
    """
    question_count = 0
    try:
        with open(config.eval_data_jsonl_file_path, "r") as file:
            for line in file:
                question_count += 1
    except FileNotFoundError as e:
        logger.error("The file does not exist: " + config.eval_data_jsonl_file_path)
        raise e

    mlflow.log_metric("question_count", question_count)

    evaluator = SpacyEvaluator(config.search_relevency_threshold)
    handler = QueryOutputHandler(config.query_data_location)
    response_generator = ResponseGenerator(
        environment, config, config.azure_oai_chat_deployment_name
    )
    for index_config in config.index_configs():
        logger.info(f"Processing index: {index_config.index_name()}")

        handler.handle_archive_by_index(
            index_config.index_name(), config.experiment_name, config.job_name
        )

        search_client = create_client(
            environment.azure_search_service_endpoint,
            index_config.index_name(),
            environment.azure_search_admin_key,
        )
        with open(config.eval_data_jsonl_file_path, "r") as file:
            with ExitStack() as stack:
                executor = stack.enter_context(
                    ThreadPoolExecutor(config.max_worker_threads)
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
