import os
import json
import shutil
import re
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
key = os.getenv("AZURE_SEARCH_ADMIN_KEY")

with open('search_config.json', 'r') as json_file:
    data = json.load(json_file)

chunk_sizes = data["chunking"]["chunk_size"]
overlap_size = data["chunking"]["overlap_size"]

embedding_dimensions = data["embedding_dimension"]
efConstructions = data["efConstruction"]
efsearchs = data["efsearch"]
name_prefix = data["name_prefix"]
search_variants = data["search_types"]
all_index_config = "generated_index_names"
chat_model_name=data["chat_model_name"]
retrieve_num_of_documents = data["retrieve_num_of_documents"]
crossencoder_model = data["crossencoder_model"]
rerank_type = data["rerank_type"]
rerank = data["rerank"]
llm_re_rank_threshold = data["llm_re_rank_threshold"]
cross_encoder_at_k = data["cross_encoder_at_k"]
temperature = data["openai_temperature"]

def query_acs(search_client, dimension, user_prompt, s_v,retrieve_num_of_documents):
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


def query_acs_multi(search_client, dimension, user_prompt, original_prompt, output_prompt, search_type, retrieve_num_of_documents, qna_context):
    context = []
    evaluation_content = output_prompt + qna_context
    for question in user_prompt:
        response = query_acs(search_client,dimension,question, search_type,retrieve_num_of_documents)

        search_response_content = evaluate_search_results(response, evaluation_content)

        if rerank == "TRUE":
            if rerank_type == "llm":
                reranked = llm_rerank_documents(search_response_content, user_prompt,chat_model_name, temperature )
                try:
                    new_docs = []
                    for key, value in reranked['documents'].items():
                        key = key.replace('document_', '')
                        numeric_data = re.findall(r'\d+\.\d+|\d+', key)
                        if int(value) > llm_re_rank_threshold:
                            new_docs.append(int(numeric_data[0]))
                        result = [search_response_content[i] for i in new_docs]
                except:
                    result = search_response_content
            elif rerank_type == "crossencoder":
                result = cross_encoder_rerank_documents(search_response_content, question,output_prompt, crossencoder_model, cross_encoder_at_k)
        else:
            result = context

        full_prompt_instruction = llm.prompts.main_prompt_instruction + "\n" + "\n".join(result)
        openai_response = generate_response(full_prompt_instruction,original_prompt,chat_model_name, temperature)
        context.append(openai_response)
        print(openai_response)

    return context

directory_path = './outputs'
if os.path.exists(directory_path) and os.path.isdir(directory_path):
    shutil.rmtree(directory_path)
os.makedirs(directory_path)

jsonl_file_path = "./eval_data.jsonl"
question_count = 0
with open(jsonl_file_path, 'r') as file:
    for line in file:
        question_count += 1

for config_item in chunk_sizes:
    for overlap in overlap_size:
        for dimension in embedding_dimensions:
            for efConstruction in efConstructions:
                for efsearch in efsearchs:
                    index_name = f"{name_prefix}-{config_item}-{overlap}-{dimension}-{efConstruction}-{efsearch}"
                    print(f"{name_prefix}-{config_item}-{overlap}-{dimension}-{efConstruction}-{efsearch}")
                    
                    search_client, index_client = create_client(service_endpoint, index_name, os.getenv("AZURE_SEARCH_ADMIN_KEY") )
                    with open(jsonl_file_path, 'r') as file:
                        for line in file:
                            data = json.loads(line)
                            user_prompt = data.get("user_prompt")
                            output_prompt = data.get("output_prompt")
                            qna_context = data.get("context")
                            for s_v in search_variants:
                                context = []
                                is_multi_question = do_we_need_multiple_questions(user_prompt,chat_model_name,temperature)
                                if re.search(r'\bHIGH\b', is_multi_question.upper()):
                                    new_questions = json.loads(we_need_multiple_questions(user_prompt,chat_model_name, temperature))
                                    new_questions['questions'].append(user_prompt)
                                    context = query_acs_multi(search_client, dimension, new_questions['questions'], user_prompt, output_prompt, s_v,retrieve_num_of_documents, qna_context)
                                else:
                                    search_response = query_acs(search_client, dimension, user_prompt, s_v,retrieve_num_of_documents)
                                    evaluation_content = user_prompt + qna_context
                                    context = evaluate_search_results(search_response, evaluation_content)

                                result = context
                                if rerank == "TRUE":
                                    if rerank_type == "llm":
                                        reranked = llm_rerank_documents(context, user_prompt,chat_model_name,temperature )
                                        try:
                                            new_docs = []
                                            for key, value in reranked['documents'].items():
                                                key = key.replace('document_', '')
                                                numeric_data = re.findall(r'\d+\.\d+|\d+', key)
                                                if int(value) > llm_re_rank_threshold:
                                                    new_docs.append(int(numeric_data[0]))

                                                result = [context[i] for i in new_docs]
                                        except:
                                            result = context
                                    elif rerank_type == "crossencoder":
                                        result = cross_encoder_rerank_documents(context,user_prompt,output_prompt,crossencoder_model,cross_encoder_at_k )
                                else:
                                    result = context


                                full_prompt_instruction = llm.prompts.main_prompt_instruction + "\n" + "\n".join(result)
                                openai_response = generate_response(full_prompt_instruction,user_prompt,chat_model_name, temperature)
                                print(openai_response)

                                output = {}
                                output["rerank"] = rerank 
                                output["rerank_type"] = rerank_type
                                output["crossencoder_model"] = crossencoder_model
                                output["llm_re_rank_threshold"] = llm_re_rank_threshold
                                output["retrieve_num_of_documents"] = retrieve_num_of_documents
                                output["cross_encoder_at_k"] = cross_encoder_at_k
                                output["question_count"] = question_count
                                output['actual'] = openai_response
                                output['expected'] = output_prompt
                                output['search_type'] = s_v 

                                write_path = f"./outputs/eval_output_{index_name}.jsonl"
                                with open(write_path, 'a') as out:
                                    json_string = json.dumps(output)
                                    out.write(json_string + "\n")
                    search_client.close()
                    index_client.close()
                    data_version = create_data_asset(write_path, index_name)
                    