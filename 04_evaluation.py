
import os
import json
from evaluation import eval
from dotenv import load_dotenv  

load_dotenv()  

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
chat_deployment_name=os.environ['CHAT_DEPLOYMENT_NAME']

experiment_name=os.environ['EXPERIMENT_NAME']

for config_item in chunk_sizes:
    for overlap in overlap_size:
        for dimension in embedding_dimensions:
            for efConstruction in efConstructions:
                for efsearch in efsearchs:
                    index_name = f"{name_prefix}-{config_item}-{overlap}-{dimension}-{efConstruction}-{efsearch}"
                    print(f"{name_prefix}-{config_item}-{overlap}-{dimension}-{efConstruction}-{efsearch}")
                    write_path = f"./outputs/eval_output_{index_name}.jsonl"
                    eval.evaluate_prompts(experiment_name, write_path )
                            
