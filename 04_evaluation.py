from dotenv import load_dotenv

load_dotenv(override=True)

from rag_experiment_accelerator.config import Config
from rag_experiment_accelerator.evaluation import eval
from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.utils.utils import get_index_name
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
import mlflow

logger = get_logger(__name__)

if __name__ == "__main__":
    config = Config()

    # This should not be global, because it complicates writing tests
    # Also usage of global variables is not recommended
    ml_client = MLClient(
        DefaultAzureCredential(),
        config.AzureMLCredentials.SUBSCRIPTION_ID,
        config.AzureMLCredentials.RESOURCE_GROUP_NAME,
        config.AzureMLCredentials.WORKSPACE_NAME,
    )
    mlflow_tracking_uri = ml_client.workspaces.get(
        ml_client.workspace_name
    ).mlflow_tracking_uri
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = mlflow.MlflowClient(mlflow_tracking_uri)

    for chunk_size in config.CHUNK_SIZES:
        for overlap in config.OVERLAP_SIZES:
            for embedding_model in config.embedding_models:
                for ef_construction in config.EF_CONSTRUCTIONS:
                    for ef_search in config.EF_SEARCHES:
                        index_name = get_index_name(
                                prefix=config.NAME_PREFIX,
                                chunk_size=chunk_size,
                                overlap=overlap,
                                embedding_model_name=embedding_model.model_name,
                                ef_construction=ef_construction,
                                ef_search=ef_search,
                            )
                        logger.info(f"Evaluating Index: {index_name}")
                        write_path = f"artifacts/outputs/eval_output_{index_name}.jsonl"
                        eval.evaluate_prompts(
                            exp_name=config.NAME_PREFIX,
                            data_path=write_path,
                            client=client,
                            chunk_size=chunk_size,
                            metric_types=config.METRIC_TYPES,
                            search_variants=config.SEARCH_VARIANTS,
                            chunk_overlap=overlap,
                            embedding_dimension=embedding_model.dimension,
                            ef_construction=ef_construction,
                            ef_search=ef_search,
                        )
                        logger.info(f"Finished evaluating index: {index_name}")
