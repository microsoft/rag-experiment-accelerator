import os
import sys
import argparse
import mlflow

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

from rag_experiment_accelerator.environment import Environment  # noqa: E402
from rag_experiment_accelerator.config import Config, Experiment  # noqa: E402
from rag_experiment_accelerator.steps.query import query_step  # noqa: E402
from rag_experiment_accelerator.paths import query_result_path  # noqa: E402


def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_data_path", type=str, help="path to the data to evaluate on"
    )
    parser.add_argument("--config_path", type=str, help="path to the config file")
    parser.add_argument(
        "--index_name_path", type=str, help="path to a file containing index name"
    )
    parser.add_argument(
        "--query_result_dir", type=str, help="path to write results of querying to"
    )
    parser.add_argument(
        "--keyvault_name", type=str, help="keyvault name to load the environment from"
    )
    args = parser.parse_args()
    with open(args.index_name_path, "r") as f:
        index_name = f.readline()

    environment = Environment.from_keyvault(args.keyvault_name)
    config = Config(environment, config_path=args.config_path)
    experiment = Experiment.from_index_name(index_name)

    with mlflow.start_run():
        mlflow_client = mlflow.MlflowClient()
        active_run_data = mlflow_client.get_run(mlflow.active_run().info.run_id).data
        print(f"Active run data: {active_run_data}")
        parent_run_id = active_run_data.tags["azureml.pipeline"]
        print(f"Parent run ID: {parent_run_id}")

    experiment_output_path = query_result_path(args.query_result_dir, experiment)
    query_step(
        experiment,
        environment,
        config,
        eval_data_path=args.eval_data_path,
        output_path=experiment_output_path,
    )


if __name__ == "__main__":
    main()
