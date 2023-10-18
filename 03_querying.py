import os
import json
import shutil
import re
from azure.search.documents import SearchClient
from config.config import Config
from dotenv import load_dotenv  
from evaluation.eval import evaluate_search_result
from evaluation.spacy_evaluator import SpacyEvaluator
from ingest_data.acs_ingest import we_need_multiple_questions, do_we_need_multiple_questions
from search_type.acs_search_methods import  (
    search_for_match_pure_vector_multi,
    search_for_match_semantic,
    search_for_match_Hybrid_multi,
    search_for_match_Hybrid_cross,
    search_for_match_text,
    search_for_match_pure_vector,
    search_for_match_pure_vector_cross,
    search_for_manual_hybrid,
    create_client
    )
from  search_type.acs_search_methods import create_client
import llm.prompts
from llm.prompt_execution import generate_response
from data_assets.data_asset import create_data_asset
from reranking.reranker import llm_rerank_documents, cross_encoder_rerank_documents


load_dotenv()  

service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")  
search_admin_key = os.getenv("AZURE_SEARCH_ADMIN_KEY")


def query_acs(search_client, dimension, user_prompt, s_v, retrieve_num_of_documents):
    if s_v == "search_for_match_semantic":
        search_response = search_for_match_semantic(search_client, dimension, user_prompt, retrieve_num_of_documents)
    elif s_v == "search_for_match_Hybrid_multi":
        search_response = search_for_match_Hybrid_multi(search_client, dimension, user_prompt, retrieve_num_of_documents)
    elif s_v == "search_for_match_Hybrid_cross":
        search_response = search_for_match_Hybrid_cross(search_client, dimension, user_prompt, retrieve_num_of_documents)
    elif s_v == "search_for_match_text":
        search_response = search_for_match_text(search_client, dimension, user_prompt, retrieve_num_of_documents)
    elif s_v == "search_for_match_pure_vector":
        search_response = search_for_match_pure_vector(search_client, dimension, user_prompt,retrieve_num_of_documents)
    elif s_v == "search_for_match_pure_vector_multi":
        search_response = search_for_match_pure_vector_multi(search_client, dimension, user_prompt,retrieve_num_of_documents)
    elif s_v == "search_for_match_pure_vector_cross":
        search_response = search_for_match_pure_vector_cross(search_client, dimension, user_prompt,retrieve_num_of_documents)
    elif s_v == "search_for_manual_hybrid":
        search_response = search_for_manual_hybrid(search_client, dimension, user_prompt,retrieve_num_of_documents)
    else:
        pass

    return search_response


def rerank_documents(
    search_response_content: list[str],
    user_prompt: str,
    output_prompt: str,
    config: Config,
):
    result = []
    if config.RERANK_TYPE == "llm":
        reranked = llm_rerank_documents(search_response_content, user_prompt, config.CHAT_MODEL_NAME, config.TEMPERATURE)
        try:
            new_docs = []
            for key, value in reranked['documents'].items():
                key = key.replace('document_', '')
                numeric_data = re.findall(r'\d+\.\d+|\d+', key)
                if int(value) > config.LLM_RERANK_THRESHOLD:
                    new_docs.append(int(numeric_data[0]))
                result = [search_response_content[i] for i in new_docs]
        except:
            result = search_response_content
    elif config.RERANK_TYPE == "crossencoder":
        result = cross_encoder_rerank_documents(search_response_content, user_prompt, output_prompt, config.CROSSENCODER_MODEL, config.CROSSENCODER_AT_K)

    return result


def query_acs_multi(
    search_client: SearchClient,
    dimension: int,
    questions: list[str],
    original_prompt: str,
    output_prompt: str,
    search_type: str,
    evaluation_content: str,
    config: Config,
    evaluator: SpacyEvaluator,
):
    context = []
    evaluations = []
    for question in questions:
        search_result = query_acs(search_client,dimension,question, search_type, config.RETRIEVE_NUM_OF_DOCUMENTS)

        docs, eval_metrics = evaluate_search_result(search_result, evaluation_content, evaluator)
        eval_metrics['query'] = question
        evaluations.append(eval_metrics)
    
        if config.RERANK:
            prompt_instruction_context = rerank_documents(docs, question, output_prompt, config)
        else:
            prompt_instruction_context = docs

        full_prompt_instruction = llm.prompts.main_prompt_instruction + "\n" + "\n".join(prompt_instruction_context)
        openai_response = generate_response(full_prompt_instruction, original_prompt, config.CHAT_MODEL_NAME, config.TEMPERATURE)
        context.append(openai_response)
        print(openai_response)

    return context, evaluations



def main(config: Config):
    directory_path = './outputs'
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path)

    jsonl_file_path = "./eval_data.jsonl"
    question_count = 0
    with open(jsonl_file_path, 'r') as file:
        for line in file:
            question_count += 1

    evaluator = SpacyEvaluator(config.SEARCH_RELEVANCY_THRESHOLD)
    for config_item in config.CHUNK_SIZES:
        for overlap in config.OVERLAP_SIZES:
            for dimension in config.EMBEDDING_DIMENSIONS:
                for efConstruction in config.EF_CONSTRUCTIONS:
                    for efsearch in config.EF_SEARCH:
                        index_name = f"{config.NAME_PREFIX}-{config_item}-{overlap}-{dimension}-{efConstruction}-{efsearch}"
                        print(f"Index: {index_name}")
                        search_client, index_client = create_client(service_endpoint, index_name, search_admin_key)
                        with open(jsonl_file_path, 'r') as file:
                            for line in file:
                                data = json.loads(line)
                                user_prompt = data.get("user_prompt")
                                output_prompt = data.get("output_prompt")
                                qna_context = data.get("context", "")

                                is_multi_question = do_we_need_multiple_questions(user_prompt, config.CHAT_MODEL_NAME, config.TEMPERATURE)
                                if is_multi_question:
                                    new_questions = json.loads(we_need_multiple_questions(user_prompt, config.CHAT_MODEL_NAME, config.TEMPERATURE))['questions']
                                    new_questions.append(user_prompt)

                                evaluation_content = user_prompt + qna_context
                                for s_v in config.SEARCH_VARIANTS:
                                    search_evaluations = []
                                    if is_multi_question:
                                        docs, search_evaluations = query_acs_multi(search_client, dimension, new_questions, user_prompt, output_prompt, s_v, evaluation_content, config, evaluator)
                                    else:
                                        search_result = query_acs(search_client, dimension, user_prompt, s_v, config.RETRIEVE_NUM_OF_DOCUMENTS)
                                        docs, evaluation = evaluate_search_result(search_result, evaluation_content, evaluator)
                                        evaluation['query'] = user_prompt
                                        search_evaluations.append(evaluation)

                                    if config.RERANK:
                                        prompt_instruction_context = rerank_documents(docs, user_prompt, output_prompt, config)
                                    else:
                                        prompt_instruction_context = docs

                                    full_prompt_instruction = llm.prompts.main_prompt_instruction + "\n" + "\n".join(prompt_instruction_context)
                                    openai_response = generate_response(full_prompt_instruction,user_prompt,config.CHAT_MODEL_NAME, config.TEMPERATURE)
                                    print(openai_response)

                                    output = {
                                        "rerank": config.RERANK,
                                        "rerank_type": config.RERANK_TYPE,
                                        "crossencoder_model": config.CROSSENCODER_MODEL,
                                        "llm_re_rank_threshold": config.LLM_RERANK_THRESHOLD,
                                        "retrieve_num_of_documents": config.RETRIEVE_NUM_OF_DOCUMENTS,
                                        "cross_encoder_at_k": config.CROSSENCODER_AT_K,
                                        "question_count": question_count,
                                        'actual': openai_response,
                                        'expected': output_prompt,
                                        'search_type': s_v,
                                        'search_evals': search_evaluations
                                    }

                                    write_path = f"./outputs/eval_output_{index_name}.jsonl"
                                    with open(write_path, 'a') as out:
                                        json_string = json.dumps(output)
                                        out.write(json_string + "\n")
                        search_client.close()
                        index_client.close()
                        data_version = create_data_asset(write_path, index_name)


if __name__ == '__main__':
    config = Config()
    main(config)