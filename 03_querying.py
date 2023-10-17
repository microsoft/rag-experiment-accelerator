import os
import json
import shutil
import re
from config.config import Config
from dotenv import load_dotenv  
from evaluation.eval import evaluate_search_results
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

# why do we pass in user_prompt and question?
# see line around 153
def rerank_documents(search_response_content, user_prompt, question, output_prompt, config: Config):
    result = []
    if config.rerank_type == "llm":
        reranked = llm_rerank_documents(search_response_content, user_prompt, config.chat_model_name, config.temperature )
        try:
            new_docs = []
            for key, value in reranked['documents'].items():
                key = key.replace('document_', '')
                numeric_data = re.findall(r'\d+\.\d+|\d+', key)
                if int(value) > config.llm_re_rank_threshold:
                    new_docs.append(int(numeric_data[0]))
                result = [search_response_content[i] for i in new_docs]
        except:
            result = search_response_content
    elif config.rerank_type == "crossencoder":
        result = cross_encoder_rerank_documents(search_response_content, question, output_prompt, config.crossencoder_model, config.cross_encoder_at_k)

    return result



def query_acs_multi(search_client, dimension, questions, original_prompt, output_prompt, search_type, config: Config):
    context = []
    search_results = []
    for question in questions:
        response = query_acs(search_client,dimension,question, search_type, config.retrieve_num_of_documents)
        search_results.append({'question': question, 'search_results': response})
        docs = []
        for r in response:
            docs.append(r['content'])
    
        if config.rerank:
            prompt_instruction_context = rerank_documents(docs, questions, question, output_prompt, config)
        else:
            prompt_instruction_context = docs

        full_prompt_instruction = llm.prompts.main_prompt_instruction + "\n" + "\n".join(prompt_instruction_context)
        openai_response = generate_response(full_prompt_instruction, original_prompt, config.chat_model_name, config.temperature)
        context.append(openai_response)
        print(openai_response)

    result = {
        'context': context,
        'search_results': search_results,
    }
    return result



def main():
    config = Config()
    directory_path = './outputs'
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path)

    jsonl_file_path = "./eval_data.jsonl"
    question_count = 0
    with open(jsonl_file_path, 'r') as file:
        for line in file:
            question_count += 1

    for config_item in config.chunk_sizes:
        for overlap in config.overlap_size:
            for dimension in config.embedding_dimensions:
                for efConstruction in config.efConstructions:
                    for efsearch in config.efsearchs:
                        index_name = f"{config.name_prefix}-{config_item}-{overlap}-{dimension}-{efConstruction}-{efsearch}"
                        print(f"{config.name_prefix}-{config_item}-{overlap}-{dimension}-{efConstruction}-{efsearch}")
                        search_client, index_client = create_client(service_endpoint, index_name, search_admin_key)
                        with open(jsonl_file_path, 'r') as file:
                            for line in file:
                                data = json.loads(line)
                                user_prompt = data.get("user_prompt")
                                output_prompt = data.get("output_prompt")
                                qna_context = data.get("context", "")
                                is_multi_question = do_we_need_multiple_questions(user_prompt, config.chat_model_name, config.temperature)
                                if is_multi_question:
                                    new_questions = json.loads(we_need_multiple_questions(user_prompt, config.chat_model_name, config.temperature))
                                    new_questions['questions'].append(user_prompt)

                                evaluation_content = user_prompt + qna_context
                                for s_v in config.search_variants:
                                    search_eval_metrics = []
                                    if is_multi_question:
                                        query_response = query_acs_multi(search_client, dimension, new_questions['questions'], user_prompt, output_prompt, s_v, config)
                                        search_eval_content = query_response['search_results']

                                    else:
                                        search_results = query_acs(search_client, dimension, user_prompt, s_v, config.retrieve_num_of_documents)
                                        search_eval_content = [{'question': user_prompt, 'search_results': search_results}]
                                    
                                    search_eval_result = evaluate_search_results(search_eval_content, evaluation_content, config.search_relevancy_threshold)
                                    search_eval_metrics.append(search_eval_result['search_metrics'])

                                    if config.rerank:
                                        prompt_instruction_context = rerank_documents(search_eval_result['content'], user_prompt, user_prompt, output_prompt, config)
                                    else:
                                        prompt_instruction_context = search_eval_result['content']

                                    full_prompt_instruction = llm.prompts.main_prompt_instruction + "\n" + "\n".join(prompt_instruction_context)
                                    openai_response = generate_response(full_prompt_instruction,user_prompt,config.chat_model_name, config.temperature)
                                    print(openai_response)

                                    output = {
                                        "rerank": config.rerank,
                                        "rerank_type": config.rerank_type,
                                        "crossencoder_model": config.crossencoder_model,
                                        "llm_re_rank_threshold": config.llm_re_rank_threshold,
                                        "retrieve_num_of_documents": config.retrieve_num_of_documents,
                                        "cross_encoder_at_k": config.cross_encoder_at_k,
                                        "question_count": question_count,
                                        'actual': openai_response,
                                        'expected': output_prompt,
                                        'search_type': s_v,
                                        'search_eval_metrics': search_eval_metrics
                                    }

                                    write_path = f"./outputs/eval_output_{index_name}2.jsonl"
                                    with open(write_path, 'a') as out:
                                        json_string = json.dumps(output)
                                        out.write(json_string + "\n")
                        search_client.close()
                        index_client.close()
                        data_version = create_data_asset(write_path, index_name)


if __name__ == '__main__':
    main()