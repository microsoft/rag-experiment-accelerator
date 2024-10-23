from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import json
import numpy as np
from azure.search.documents import SearchClient
from dotenv import load_dotenv
import mlflow
from openai import BadRequestError

from sklearn.metrics.pairwise import cosine_similarity

from rag_experiment_accelerator.artifact.handlers.query_output_handler import (
    QueryOutputHandler,
)
from rag_experiment_accelerator.artifact.models.query_output import QueryOutput
from rag_experiment_accelerator.checkpoint import cache_with_checkpoint
from rag_experiment_accelerator.config.config import Config
from rag_experiment_accelerator.config.index_config import IndexConfig
from rag_experiment_accelerator.embedding.embedding_model import EmbeddingModel
from rag_experiment_accelerator.evaluation.search_eval import (
    evaluate_search_result,
)
from rag_experiment_accelerator.evaluation.spacy_evaluator import (
    SpacyEvaluator,
)

from rag_experiment_accelerator.ingest_data.acs_ingest import (
    do_we_need_multiple_questions,
    generate_multiple_questions,
)
from rag_experiment_accelerator.nlp.preprocess import Preprocess
from rag_experiment_accelerator.rag_cache.rag_cache import RagCache
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
        preprocess: bool = False,
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
        preprocess=preprocess,
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
    match config.rerank.type:
        case "llm":
            return llm_rerank_documents(
                docs,
                user_prompt,
                response_generator,
                config.rerank.llm_rerank_threshold,
            )
        case "cross_encoder":
            return cross_encoder_rerank_documents(
                docs,
                user_prompt,
                output_prompt,
                config.rerank.cross_encoder_model,
                config.rerank.cross_encoder_at_k,
            )
        case _:
            return []


def hyde(
        config: Config,
        response_generator: ResponseGenerator,
        queries: list[str],
):
    if config.query_expansion.hyde == "disabled":
        return queries

    hyde_prompt = {
        "generated_hypothetical_answer": prompt_generate_hypothetical_answer,
        "generated_hypothetical_document_to_answer": prompt_generate_hypothetical_document,
    }

    if config.query_expansion.hyde not in hyde_prompt:
        raise NotImplementedError(
            f"configuration for hyde with value of [{config.query_expansion.hyde}] is not supported"
        )

    generated_queries = [
        response_generator.generate_response(
            hyde_prompt[config.query_expansion.hyde], text=query
        )
        for query in queries
    ]
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
        config.query_expansion.min_query_expansion_related_question_similarity_score,
    )

    return questions


def deduplicate_search_results(search_results: list[dict]) -> list[dict]:
    doc_set = set()
    score_dict = {}
    doc_ids = []

    # deduplicate and sort retrieved documents by using a set
    for doc in search_results:
        doc_set.add(doc["content"])  # Create a tuple of content and id
        doc_ids.append(doc["id"])
        score_dict[doc["content"]] = doc["@search.score"]

    search_result = list(doc_set)
    search_result = [
        {"content": doc, "@search.score": score_dict[doc]} for doc in search_result
    ]
    search_result.sort(key=lambda x: x["@search.score"], reverse=True)

    return search_result, doc_ids


class QueryAndEvalACSResult:
    def __init__(self, documents: list[str], evaluations: dict[str, any], doc_ids: list[str] = None):
        self.documents = documents
        self.evaluations = evaluations
        self.doc_ids = doc_ids


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
        preprocess: bool = False,
) -> QueryAndEvalACSResult:
    """
    Queries the Azure AI Search service using the provided search client and parameters, and evaluates the search
    results using the provided evaluator and evaluation content. Returns a QueryAndEvalACSResult object containing
    the retrieved documents and the evaluation results.

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
        QueryAndEvalACSResult: An object containing the retrieved documents and the evaluation results.
    """

    if config.query_expansion.query_expansion:
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
            preprocess=preprocess,
        )
        search_results.extend(search_result)

    search_results, doc_ids = deduplicate_search_results(search_results)
    search_result = search_result[: config.search.retrieve_num_of_documents]

    docs, evaluation = evaluate_search_result(
        search_results, evaluation_content, evaluator
    )

    evaluation["query"] = query
    return QueryAndEvalACSResult(docs, evaluation, doc_ids)


def filter_non_related_questions(
        query,
        generated_questions,
        embedding_model,
        min_query_expansion_related_question_similarity_score,
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
        if similarity_score >= min_query_expansion_related_question_similarity_score:
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
        preprocess: bool = False,
) -> QueryAndEvalACSResult:
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
        QueryAndEvalACSResult: : An object containing the retrieved documents and the evaluation results for each question.
    """
    context = []
    evaluations = []

    for question in questions:
        result = query_and_eval_acs(
            search_client=search_client,
            embedding_model=embedding_model,
            query=question,
            search_type=search_type,
            evaluation_content=evaluation_content,
            retrieve_num_of_documents=config.search.retrieve_num_of_documents,
            evaluator=evaluator,
            config=config,
            response_generator=response_generator,
            preprocess=preprocess,
        )
        if len(result.documents) == 0:
            logger.warning(f"No documents found for question: {question}")
            continue

        evaluations.append(result.evaluations)

        if config.rerank.enabled:
            prompt_instruction_context = rerank_documents(
                result.documents, question, output_prompt, config, response_generator
            )
        else:
            prompt_instruction_context = result.documents

        # TODO: Here was a bug, caused by the fact that we are not limiting the number of documents to retrieve
        # Current solution is just forcefully limiting the number of documents to retrieve assuming they are sorted
        if len(prompt_instruction_context) > config.search.retrieve_num_of_documents:
            prompt_instruction_context = prompt_instruction_context[
                                         : config.search.retrieve_num_of_documents
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

    return QueryAndEvalACSResult(documents=context, evaluations=evaluations)


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
        rag_cache: RagCache,
):
    logger.info(f"Processing question {line_number + 1} out of {question_count}\n\n")
    data: dict[str, any] = json.loads(line)
    user_prompt = data.get("user_prompt")
    output_prompt = data.get("output_prompt")
    qna_context = data.get("context", "")

    is_multi_question = (
            config.query_expansion.expand_to_multiple_questions
            and do_we_need_multiple_questions(user_prompt, response_generator, config)
    )

    new_questions = []
    if is_multi_question:
        new_questions = generate_multiple_questions(user_prompt, response_generator)

        if new_questions is None:
            logger.warning(
                f"Unable to generate multiple questions for: {user_prompt}. Skipping..."
            )
            is_multi_question = False
        else:
            new_questions.append(user_prompt)

    evaluation_content = user_prompt + qna_context

    embedding_model1 = config.get_embedding_model(
        index_config.embedding_model.model_name
    )

    # pre_process = Preprocess(enabled=index_config.chunking.preprocess)
    # embedding = embedding_model1.generate_embedding(chunk=pre_process.preprocess(user_prompt))
    # results = None

    try:
        # if environment.rag_global_cache:
        #     # print("== calling prompt cache")
        #     print(f"\n Query to LLM: {user_prompt} ")
        #     results = rag_cache.getFromRagCache(embedding)
        #     if results is not None and results:
        #         print(f"Prompt cache Response:", results["content"])
        #
        # if results is None or not results:
        for s_v in config.search.search_type:
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
                rag_cache
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


# @cache_with_checkpoint(
#     id="user_prompt+output_prompt+qna_context+index_config.index_name()"
# )
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
        rag_cache=None,
):
    search_evals = []

    response_generator = ResponseGenerator(
        environment, config, config.openai.azure_oai_chat_deployment_name
    )

    embedding_model = config.get_embedding_model(
        index_config.embedding_model.model_name
    )

    if is_multi_question:
        result = query_and_eval_acs_multi(
            search_client=search_client,
            embedding_model=embedding_model,
            questions=new_questions,
            original_prompt=user_prompt,
            output_prompt=output_prompt,
            search_type=s_v,
            evaluation_content=evaluation_content,
            config=config,
            evaluator=evaluator,
            response_generator=response_generator,
            preprocess=index_config.chunking.preprocess,
        )
    else:
        result = query_and_eval_acs(
            search_client=search_client,
            embedding_model=embedding_model,
            query=user_prompt,
            search_type=s_v,
            evaluation_content=evaluation_content,
            retrieve_num_of_documents=config.search.retrieve_num_of_documents,
            evaluator=evaluator,
            config=config,
            response_generator=response_generator,
            preprocess=index_config.chunking.preprocess,
        )
        search_evals.append(result.evaluations)
    if config.rerank.enabled and len(result.documents) > 0:
        prompt_instruction_context = rerank_documents(
            result.documents,
            user_prompt,
            output_prompt,
            config,
            response_generator,
        )
    else:
        prompt_instruction_context = result.documents

    pre_process = Preprocess(enabled=index_config.chunking.preprocess)
    user_prompt_embedding = embedding_model.generate_embedding(chunk=pre_process.preprocess(user_prompt))
    cached_response = None

    if rag_cache is not None:
        # Check if result is available in cache
        cached_response = rag_cache.getFromRagCache(user_prompt_embedding)

    if cached_response:
        logger.info("**** Reading from RAG cache instead of making an OpenAI call ****")
        logger.info(" cached response", cached_response)
        openai_response = cached_response
    else:
        logger.info("**** Cache miss or no cache, making OpenAI call ****")
        # Generate response using OpenAI
        openai_response = response_generator.generate_response(
            main_instruction,
            context="\n".join(prompt_instruction_context),
            question=user_prompt,
        )

        # Add to cache if doc IDs exist
        doc_ids_str = ""
        if result.doc_ids is not None:
            doc_ids_str = ", ".join(result.doc_ids)

        knowledge_base_docIds = doc_ids_str
        logger.info("\n Adding to cache")
        if rag_cache is not None:
            rag_cache.addToCache(user_prompt, user_prompt_embedding, openai_response, knowledge_base_docIds)



    output = QueryOutput(
        rerank=config.rerank.enabled,
        rerank_type=config.rerank.type,
        cross_encoder_model=config.rerank.cross_encoder_model,
        llm_rerank_threshold=config.rerank.llm_rerank_threshold,
        retrieve_num_of_documents=config.search.retrieve_num_of_documents,
        cross_encoder_at_k=config.rerank.cross_encoder_at_k,
        question_count=question_count,
        actual=openai_response,
        expected=output_prompt,
        search_type=s_v,
        search_evals=search_evals,
        context=qna_context,
        retrieved_contexts=prompt_instruction_context,
        question=user_prompt
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
        with open(config.path.eval_data_file, "r") as file:
            for line in file:
                question_count += 1
    except FileNotFoundError as e:
        logger.error("The file does not exist: " + config.path.eval_data_file)
        raise e

    mlflow.log_metric("question_count", question_count)

    evaluator = SpacyEvaluator(config.search.search_relevancy_threshold)
    handler = QueryOutputHandler(config.path.query_data_dir)
    response_generator = ResponseGenerator(
        environment, config, config.openai.azure_oai_chat_deployment_name
    )
    for index_config in config.index.flatten():
        index_name = index_config.index_name()
        logger.info(f"Processing index: {index_name}")

        handler.handle_archive_by_index(
            index_name, config.experiment_name, config.job_name
        )

        search_client = create_client(
            environment.azure_search_service_endpoint,
            index_name,
            environment.azure_search_admin_key,
        )

        if environment.rag_global_cache:
            rag_cache = RagCache(
                environment=environment,
                config=config,
                index_name=index_name
            )
            rag_cache.create_rag_cache_index()

        with open(config.path.eval_data_file, "r") as file:
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
                        rag_cache
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
