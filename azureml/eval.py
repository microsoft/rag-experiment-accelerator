import os
import sys
import argparse
import mlflow
from mlflow import MlflowClient


project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

from rag_experiment_accelerator.config.environment import Environment  # noqa: E402
from rag_experiment_accelerator.config.config import Config  # noqa: E402
from rag_experiment_accelerator.config.index_config import IndexConfig  # noqa: E402
from rag_experiment_accelerator.run.evaluation import run as eval_run  # noqa: E402


def _get_parent_mlflow_run_id(mlflow_client: MlflowClient):
    """
    The MLFlow run will be already started by the parent pipeline,
    retrieve the run_id to collect metrics into the parent run
    """
    with mlflow.start_run():
        mlflow_run = mlflow_client.get_run(mlflow.active_run().info.run_id)
        return mlflow_run.data.tags["azureml.pipeline"]


def main():
    """Main function of the script."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, help="input: path to the config file"
    )
    parser.add_argument(
        "--index_name_path",
        type=str,
        help="input: path to a file containing index name",
    )
    parser.add_argument(
        "--query_result_dir",
        type=str,
        help="input: path to read results of querying from",
    )
    parser.add_argument(
        "--keyvault", type=str, help="keyvault to load the environment from"
    )
    parser.add_argument(
        "--eval_result_dir",
        type=str,
        help="output: path to write results of evalulation to",
    )
    args = parser.parse_args()

    environment = Environment.from_keyvault(args.keyvault)
    config = Config(environment, config_path=args.config_path)
    with open(args.index_name_path, "r") as f:
        index_name = f.readline()
    index_config = IndexConfig.from_index_name(index_name, config)

    config.QUERY_DATA_LOCATION = args.query_result_dir
    config.EVAL_DATA_LOCATION = args.eval_result_dir

    mlflow_client = mlflow.MlflowClient()
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    with mlflow.start_run(run_id=_get_parent_mlflow_run_id(mlflow_client)):
        eval_run(
            environment, config, index_config, mlflow_client, name_suffix="_result"
        )


if __name__ == "__main__":
    main()
