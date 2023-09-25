import os
import json
import shutil
import re
from dotenv import load_dotenv  
from init_Index.create_index import create_acs_index
from doc_loader.pdfLoader import load_pdf_files
from embedding.gen_embeddings import generate_embedding
from ingest_data.acs_ingest import upload_data, we_need_multiple_questions, do_we_need_multiple_questions
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
from reranking.llm_reranker import rerank_documents


load_dotenv()  
experiment_name=os.environ['EXPERIMENT_NAME']

service_endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")  
key = os.getenv("AZURE_SEARCH_ADMIN_KEY")

with open('search_config.json', 'r') as json_file:
    data = json.load(json_file)

chunk_sizes = data["chunking"]["chunk_size"]
overall_size = data["chunking"]["overall_size"]

embedding_dimensions = data["embedding_dimension"]
efConstructions = data["efConstruction"]
efsearchs = data["efsearch"]
name_prefix = data["name_prefix"]
search_variants = data["search_types"]
all_index_config = "generated_index_names"
chat_deployment_name=os.environ['CHAT_DEPLOYMENT_NAME']

rerank = data["rerank"]
re_rank_threshold = data["re_rank_threshold"]

def query_acs(search_client, dimension, user_prompt):
    if s_v == "search_for_match_semantic":
        search_response = search_for_match_semantic(search_client, dimension, user_prompt)
    elif s_v == "search_for_match_Hybrid_multi":
        search_response = search_for_match_Hybrid_multi(search_client, dimension, user_prompt)
    elif s_v == "search_for_match_Hybrid_cross":
        search_response = search_for_match_Hybrid_cross(search_client, dimension, user_prompt)
    elif s_v == "search_for_match_text":
        search_response = search_for_match_text(search_client, dimension, user_prompt)
    elif s_v == "search_for_match_pure_vector":
        search_response = search_for_match_pure_vector(search_client, dimension, user_prompt)
    elif s_v == "search_for_match_pure_vector_multi":
        search_response = search_for_match_pure_vector_multi(search_client, dimension, user_prompt)
    elif s_v == "search_for_match_pure_vector_cross":
        search_response = search_for_match_pure_vector_cross(search_client, dimension, user_prompt)
    elif s_v == "search_for_manual_hybrid":
        search_response = search_for_manual_hybrid(search_client, dimension, user_prompt)
    else:
        pass

    return search_response


def query_acs_multi(search_client, dimension, user_prompt, original_prompt):
    context = []
    for question in user_prompt:
        search_response = []
        response = query_acs(search_client,dimension,question)

        for result in response:  
            print(f"Score: {result['@search.score']}")  
            print(f"Content: {result['content']}")  
            print("++++++++++++++++++++++++++++++++++")
            search_response.append(result['content'])

        if rerank == "TRUE":
            reranked = rerank_documents(search_response, original_prompt,chat_deployment_name )
            try:
                new_docs = []
                for key, value in reranked['documents'].items():
                    key = key.replace('document_', '')
                    numeric_data = re.findall(r'\d+\.\d+|\d+', key)
                    if int(value) > re_rank_threshold:
                        new_docs.append(int(numeric_data[0]))

                result = [search_response[i] for i in new_docs]
            except:
                result = search_response
        else:
            result = search_response

        full_prompt_instruction = llm.prompts.main_prompt_instruction + "\n" + "\n".join(result)
        openai_response = generate_response(full_prompt_instruction,original_prompt,chat_deployment_name)
        context.append(openai_response)
        print(openai_response)

    return context

directory_path = './outputs'
if os.path.exists(directory_path) and os.path.isdir(directory_path):
    shutil.rmtree(directory_path)
os.makedirs(directory_path)

for config_item in chunk_sizes:
    for overlap in overall_size:
        for dimension in embedding_dimensions:
            for efConstruction in efConstructions:
                for efsearch in efsearchs:
                    index_name = f"{name_prefix}-{config_item}-{overlap}-{dimension}-{efConstruction}-{efsearch}"
                    print(f"{name_prefix}-{config_item}-{overlap}-{dimension}-{efConstruction}-{efsearch}")
                    
                    search_client, index_client = create_client(service_endpoint, index_name, os.getenv("AZURE_SEARCH_ADMIN_KEY") )
                    with open("./eval_data.jsonl", 'r') as file:
                        for line in file:
                            data = json.loads(line)
                            user_prompt = data.get("user_prompt")
                            output_prompt = data.get("output_prompt")
                            for s_v in search_variants:
                                context = []
                                is_multi_question = do_we_need_multiple_questions(user_prompt)
                                if re.search(r'\bHIGH\b', is_multi_question):
                                    try:
                                        new_questions = json.loads(we_need_multiple_questions(user_prompt))
                                        new_questions['questions'].append(user_prompt)
                                    except:
                                        new_questions = json.loads(we_need_multiple_questions(user_prompt))
                                        new_questions['questions'].append(user_prompt)
                                    context = query_acs_multi(search_client, dimension, new_questions['questions'], user_prompt)
                                    result = context
                                else:
                                    search_response = query_acs(search_client, dimension, user_prompt)
                                    for result in search_response:  
                                        print(f"Score: {result['@search.score']}")  
                                        print(f"Content: {result['content']}")  
                                        print("++++++++++++++++++++++++++++++++++")
                                        context.append(result['content'])

                                if rerank == "TRUE":
                                    reranked = rerank_documents(context, user_prompt,chat_deployment_name )
                                    try:
                                        new_docs = []
                                        for key, value in reranked['documents'].items():
                                            key = key.replace('document_', '')
                                            numeric_data = re.findall(r'\d+\.\d+|\d+', key)
                                            if int(value) > re_rank_threshold:
                                                new_docs.append(int(numeric_data[0]))

                                            esult = [context[i] for i in new_docs]
                                    except:
                                        result = context
                                else:
                                    result = context


                                full_prompt_instruction = llm.prompts.main_prompt_instruction + "\n" + "\n".join(result)
                                openai_response = generate_response(full_prompt_instruction,user_prompt,chat_deployment_name)
                                print(openai_response)

                    # here
                                output = {}
                                output["query_type"] = s_v # specific
                                output['actual'] = openai_response
                                output['expected'] = output_prompt
                                output['search_type'] = s_v # specific

                                write_path = f"./outputs/eval_output_{index_name}.jsonl"
                                with open(write_path, 'a') as out:
                                    json_string = json.dumps(output)
                                    out.write(json_string + "\n")
                    search_client.close()
                    index_client.close()
                    data_version = create_data_asset(write_path, index_name)
                    