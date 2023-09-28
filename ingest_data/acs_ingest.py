

import json
import hashlib
import json
from azure.core.credentials import AzureKeyCredential  
from azure.search.documents import SearchClient  
import llm.prompts
from llm.prompt_execution import generate_response
from embedding.gen_embeddings import generate_embedding
from nlp.preprocess import Preprocess
import pandas as pd

pre_process = Preprocess()

def my_hash(s):
    return hashlib.md5(s.encode()).hexdigest()


def generate_title(chunk):
    response = generate_response(llm.prompts.prompt_instruction_title,chunk,"gpt-35-turbo")
    return response

def generate_summary(chunk):
    response = generate_response(llm.prompts.prompt_instruction_summary,chunk,"gpt-35-turbo")
    return response

def upload_data(chunks, service_endpoint, index_name, search_key, dimension):
    credential = AzureKeyCredential(search_key)
    search_client = SearchClient(endpoint=service_endpoint, index_name=index_name, credential=credential)
    documents = []
    for i, chunk in enumerate(chunks):
        input_data = {}
        input_data['id'] = str(my_hash(chunk["content"]))
        title = generate_title(str(chunk["content"]))
        summary = generate_summary(str(chunk["content"]))
        input_data['title'] = title
        input_data['summary'] = summary
        input_data['content'] = str(chunk["content"])
        input_data['filename'] = "test"
        input_data['contentVector'] = chunk["content_vector"][0]
        input_data['contentSummary'] = generate_embedding(dimension,str(pre_process.preprocess(summary)))[0]
        input_data['contentTitle'] =  generate_embedding(dimension,str(pre_process.preprocess(title)))[0]
        documents.append(input_data)

    results = search_client.upload_documents(documents)  
    print(f"Uploaded {len(documents)} documents") 
    print("done")


def generate_qna(docs):
    column_names = ['user_prompt', 'output_prompt', 'context']
    new_df = pd.DataFrame(columns=column_names)
    for i, chunk in enumerate(docs):
        if len(chunk.page_content) > 50:
            response = generate_response(llm.prompts.generate_qna_instruction, chunk.page_content, "gpt-35-turbo")
            try:
                response_dict = json.loads( response )
                for each_pair in response_dict["prompts"]:
                    data = {
                            'user_prompt': each_pair["question"],
                            'output_prompt': each_pair["answer"],
                            'context': chunk.page_content
                    }
            except:
                print("json_string2 is not valid JSON")
            new_df = new_df._append(data, ignore_index=True)
    new_df.to_json("eval_data.jsonl", orient='records', lines=True)

def we_need_multiple_questions(question):
    full_prompt_instruction = llm.prompts.multiple_prompt_instruction + "\n"+  "question: "  + question + "\n"
    response1= generate_response(full_prompt_instruction,"","gpt-35-turbo")
    return response1

def do_we_need_multiple_questions(question):
    full_prompt_instruction = llm.prompts.do_need_multiple_prompt_instruction + "\n"+  "question: "  + question + "\n"
    response1= generate_response(full_prompt_instruction,"","gpt-35-turbo")
    return response1