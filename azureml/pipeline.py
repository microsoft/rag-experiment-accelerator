import os
import sys
import argparse

from azure.ai.ml import MLClient, Input, Output, dsl
from azure.ai.ml.parallel import parallel_run_function, RunFunction
import azure.ai.ml.entities
from azure.identity import DefaultAzureCredential

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)

from rag_experiment_accelerator.config.environment import Environment  # noqa: E402
from rag_experiment_accelerator.config import Config  # noqa: E402
from rag_experiment_accelerator.config.index_config import IndexConfig  # noqa: E402
from rag_experiment_accelerator.config.paths import mlflow_run_name  # noqa: E402


AML_ENVIRONMENT_NAME = "rag-env"
INDEX_STEP_RETRIES = 3
INDEX_STEP_TIMEOUT_SECONDS = 10 * 3600
INDEX_STEP_FILES_PER_BATCH = 4
INDEX_STEP_CONCURRENCY_PER_INSTANCE = 4
INDEX_STEP_ERROR_THRESHOLD = 1


def generate_conda_file():
    """Generates a file describing a conda environment compatible with AML."""
    conda_config = """name: aml-env
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pkg-config
  - cmake
  - spacy
  - spacy-model-en_core_web_lg==3.5.0
  - pip:
"""
    with open("requirements.txt", "r") as requirements_file:
        pip_dependencies = requirements_file.readlines()
    conda_config += "".join(["    - " + dependency for dependency in pip_dependencies])

    conda_filename = "conda.generated.yaml"
    with open(conda_filename, "w") as conda_file:
        conda_file.write(conda_config)
    return conda_filename


def start_pipeline(
    index_config: IndexConfig,
    config: Config,
    config_dir: str,
    config_path: str,
    environment: Environment,
):
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=environment.aml_subscription_id,
        resource_group_name=environment.aml_resource_group_name,
        workspace_name=environment.aml_workspace_name,
    )

    # Generate conda file
    conda_filename = generate_conda_file()

    # Create environment for AML
    pipeline_job_env = azure.ai.ml.entities.Environment(
        name=AML_ENVIRONMENT_NAME,
        description="Environment for RAG Experiment Accelerator",
        conda_file=conda_filename,
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    )
    pipeline_job_env = ml_client.environments.create_or_update(pipeline_job_env)

    index_pipeline_component = parallel_run_function(
        name="index_job",
        display_name=f"Index documents for the experiment {index_config.index_name()}",
        description="Upload input documents for RAG accelerator into Azure Search Index",
        inputs={
            "data": Input(type="uri_folder"),
            "config_dir": Input(type="uri_folder"),
        },
        outputs={"index_name": Output(type="uri_file", mode="rw_mount")},
        input_data="${{inputs.data}}",
        instance_count=int(environment.aml_compute_instances_number),
        max_concurrency_per_instance=INDEX_STEP_CONCURRENCY_PER_INSTANCE,
        mini_batch_size=str(INDEX_STEP_FILES_PER_BATCH),
        mini_batch_error_threshold=INDEX_STEP_ERROR_THRESHOLD,
        retry_settings=dict(
            max_retries=INDEX_STEP_RETRIES, timeout=INDEX_STEP_TIMEOUT_SECONDS
        ),
        task=RunFunction(
            code="./",
            entry_script="azureml/index.py",
            program_arguments="""--data_dir ${{inputs.data}} \
                --config_dir ${{inputs.config_dir}}
                --index_name_path ${{outputs.index_name}}"""
            + f" --keyvault_name {environment.keyvault_name}"
            + f" --config_path {config_path}"
            + f" --index_name {index_config.index_name()}",
            environment=pipeline_job_env,
            append_row_to="${{outputs.index_name}}",
        ),
    )

    @dsl.pipeline(
        name=mlflow_run_name(config),
        compute=environment.aml_compute_name,
        description="RAG Experiment Pipeline",
    )
    def rag_pipeline(config_dir_input, data_input, eval_data_input):
        index_job = index_pipeline_component(
            data=data_input, config_dir=config_dir_input
        )

        return {
            # TODO: this is temporary, in the future the pipeline
            # will return the result of the eval step
            "index_result": index_job.outputs.index_name
        }

    # Save the environment into Keyvault for the pipeline steps to retrieve later
    environment.to_keyvault()

    pipeline = rag_pipeline(
        config_dir_input=Input(type="uri_file", path=config_dir),
        data_input=Input(type="uri_folder", path=config.data_dir),
        # eval_data_input=Input(type="uri_file", path=config.EVAL_DATA_JSON_FILE_PATH),
    )
    ml_client.jobs.create_or_update(pipeline, experiment_name=config.NAME_PREFIX)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_dir", type=str, help="path to the config directory", default="."
    )
    parser.add_argument(
        "--data_dir", type=str, help="path to the data folder", default="./data"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="relative path to the config file",
        default="config.json",
    )
    args = parser.parse_args()

    environment = Environment.from_env()
    config = Config(args.config_dir, args.data_dir, args.config_path)
    # Starting multiple pipelines hence unable to stream them
    for index_config in config.index_configs():
        start_pipeline(
            index_config, config, args.config_dir, args.config_path, environment
        )
