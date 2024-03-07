import os
import sys
import argparse
import mlflow

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

from rag_experiment_accelerator.environment import Environment  # noqa: E402
from rag_experiment_accelerator.config import Config, Experiment  # noqa: E402
from rag_experiment_accelerator.steps.eval import eval_step  # noqa: E402
from rag_experiment_accelerator.paths import query_result_path  # noqa: E402


def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="path to the config file")
    parser.add_argument(
        "--index_name_path", type=str, help="path to a file containing index name"
    )
    parser.add_argument(
        "--query_result_dir", type=str, help="path to read results of querying from"
    )
    parser.add_argument(
        "--eval_result_dir", type=str, help="path to write results of evalulation to"
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
    mlflow_client = mlflow.MlflowClient()

    mlflow.set_experiment(config.EXPERIMENT_NAME)
    # The MLFlow run will be already started by the parent pipeline,
    # retrieve the run_id to collect metrics into the parent run
    with mlflow.start_run():
        mlflow_run = mlflow_client.get_run(mlflow.active_run().info.run_id)
        parent_run_id = mlflow_run.data.tags["azureml.pipeline"]

    with mlflow.start_run(run_id=parent_run_id):
        eval_step(
            experiment=experiment,
            config=config,
            data_path=query_result_path(args.query_result_dir, experiment),
            output_dir=args.eval_result_dir,
            mlflow_client=mlflow_client,
        )


if __name__ == "__main__":
    main()
