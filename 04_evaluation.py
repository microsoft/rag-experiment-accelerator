from dotenv import load_dotenv
load_dotenv(override=True)

from rag_experiment_accelerator.utils.auth import get_default_az_cred
from rag_experiment_accelerator.config import Config
from rag_experiment_accelerator.evaluation import eval
from rag_experiment_accelerator.utils.logging import get_logger
from azure.ai.ml import MLClient
import mlflow

logger = get_logger(__name__)

def main():
    config = Config()

    ml_client = MLClient(
        get_default_az_cred(),
        config.AzureMLCredentials.SUBSCRIPTION_ID,
        config.AzureMLCredentials.RESOURCE_GROUP_NAME,
        config.AzureMLCredentials.WORKSPACE_NAME,
    )
    mlflow_tracking_uri = ml_client.workspaces.get(
        ml_client.workspace_name
    ).mlflow_tracking_uri
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = mlflow.MlflowClient(mlflow_tracking_uri)

    for config_item in config.CHUNK_SIZES:
        for overlap in config.OVERLAP_SIZES:
            for dimension in config.EMBEDDING_DIMENSIONS:
                for ef_construction in config.EF_CONSTRUCTIONS:
                    for ef_search in config.EF_SEARCHES:
                        index_name = f"{config.NAME_PREFIX}-{config_item}-{overlap}-{dimension}-{ef_construction}-{ef_search}"
                        logger.info(
                            f"{config.NAME_PREFIX}-{config_item}-{overlap}-{dimension}-{ef_construction}-{ef_search}"
                        )
                        write_path = f"artifacts/outputs/eval_output_{index_name}.jsonl"
                        eval.evaluate_prompts(
                            exp_name=config.NAME_PREFIX,
                            data_path=write_path,
                            config=config,
                            client=client,
                            chunk_size=config_item,
                            chunk_overlap=overlap,
                            embedding_dimension=dimension,
                            ef_construction=ef_construction,
                            ef_search=ef_search,
                        )


if __name__ == "__main__":
    main()
