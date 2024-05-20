import os
import sys
import argparse
from azure.ai.ml import MLClient, Input, Output, dsl, command
from azure.ai.ml.parallel import parallel_run_function, RunFunction
import azure.ai.ml.entities
from azure.ai.ml.entities import Job
import mlflow
import warnings


from rag_experiment_accelerator.config.paths import (
    formatted_datetime_suffix,
)  # noqa: E402
from rag_experiment_accelerator.config.environment import Environment  # noqa: E402
from rag_experiment_accelerator.config.config import (
    Config,
    ExecutionEnvironment,
)  # noqa: E402
from rag_experiment_accelerator.config.index_config import IndexConfig  # noqa: E402
from rag_experiment_accelerator.utils.auth import get_default_az_cred  # noqa: E402
from rag_experiment_accelerator.utils.logging import get_logger  # noqa: E402


project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_dir)
logger = get_logger(__name__)
AML_ENVIRONMENT_NAME = "CliV2AnonymousEnvironment"
INDEX_STEP_RETRIES = 3
INDEX_STEP_TIMEOUT_SECONDS = 10 * 3600
INDEX_STEP_FILES_PER_BATCH = 4
INDEX_STEP_ERROR_THRESHOLD = 1


def generate_conda_file():
    """Generates a file describing a conda environment compatible with AML."""
    conda_config = """name: aml-env
channels:
  - conda-forge
dependencies:
  - python=3.11
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


def initialise_mlflow_client(environment: Environment, config: Config):
    """
    Initializes the ML client and sets the MLflow tracking URI.
    """
    ml_client = MLClient(
        get_default_az_cred(),
        environment.aml_subscription_id,
        environment.aml_resource_group_name,
        environment.aml_workspace_name,
    )
    mlflow_tracking_uri = ml_client.workspaces.get(
        ml_client.workspace_name
    ).mlflow_tracking_uri
    mlflow.set_tracking_uri(mlflow_tracking_uri)

    return mlflow.MlflowClient(mlflow_tracking_uri)


def start_pipeline(
    environment: Environment,
    config: Config,
    index_config: IndexConfig,
    config_path: str,
    mlflow_client: mlflow.MlflowClient,
) -> Job:
    warnings.filterwarnings("ignore", category=UserWarning, module="azure.ai.ml")

    ml_client = MLClient(
        credential=get_default_az_cred(),
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
        experiment_name=config.experiment_name,
        display_name=f"Index documents for {config.experiment_name} experiment",
        description="Preprocess documents, split into smaller chunks, embed and enrich them, and finally upload documents chunks for retrieval into Azure Search Index",
        inputs={
            "data": Input(type="uri_folder"),
            "config_path": Input(type="uri_file"),
            "mlflow_tracking_uri": Input(type="string"),
        },
        outputs={"index_name": Output(type="uri_file", mode="rw_mount")},
        input_data="${{inputs.data}}",
        instance_count=int(environment.aml_compute_instances_number),
        mini_batch_size=str(INDEX_STEP_FILES_PER_BATCH),
        mini_batch_error_threshold=INDEX_STEP_ERROR_THRESHOLD,
        retry_settings=dict(
            max_retries=INDEX_STEP_RETRIES, timeout=INDEX_STEP_TIMEOUT_SECONDS
        ),
        task=RunFunction(
            code="./",
            entry_script="azureml/index.py",
            program_arguments="""--data_dir ${{inputs.data}} \
                --index_name_path ${{outputs.index_name}} \
                --config_path ${{inputs.config_path}} \
                --mlflow_tracking_uri ${{inputs.mlflow_tracking_uri}}"""
            + f" --keyvault {environment.azure_key_vault_endpoint}"
            + f" --index_name {index_config.index_name()}",
            environment=pipeline_job_env,
            append_row_to="${{outputs.index_name}}",
        ),
        environment_variables={
            "LOGGING_LEVEL": os.getenv("LOGGING_LEVEL", ""),
            "MAX_WORKER_THREADS": os.getenv("MAX_WORKER_THREADS", ""),
        },
    )

    query_pipeline_component = command(
        name="query_job",
        experiment_name=config.experiment_name,
        display_name="Query documents for the experiment",
        description="Query documents for the experiment",
        inputs={
            "index_name": Input(type="uri_file"),
            "config_path": Input(type="uri_file"),
            "eval_data": Input(type="uri_file"),
            "mlflow_tracking_uri": Input(type="string"),
        },
        outputs={"query_result": Output(type="uri_folder", mode="rw_mount")},
        code="./",
        command="""python ./azureml/query.py \
            --eval_data_path ${{inputs.eval_data}} \
            --config_path ${{inputs.config_path}} \
            --index_name_path ${{inputs.index_name}} \
            --query_result_dir ${{outputs.query_result}} \
            --mlflow_tracking_uri ${{inputs.mlflow_tracking_uri}}"""
        + f" --keyvault {environment.azure_key_vault_endpoint}",
        environment=pipeline_job_env,
        environment_variables={
            "LOGGING_LEVEL": os.getenv("LOGGING_LEVEL", ""),
            "MAX_WORKER_THREADS": os.getenv("MAX_WORKER_THREADS", ""),
        },
    )

    eval_pipeline_component = command(
        name="eval_job",
        experiment_name=config.experiment_name,
        display_name="Evaluate experiment",
        description="Evaluate experiment",
        inputs={
            "index_name": Input(type="uri_file"),
            "config_path": Input(type="uri_file"),
            "query_result": Input(type="uri_folder"),
            "mlflow_tracking_uri": Input(type="string"),
        },
        outputs=dict(eval_result=Output(type="uri_folder", mode="rw_mount")),
        code="./",
        command="""python ./azureml/eval.py \
                --config_path ${{inputs.config_path}} \
                --index_name_path ${{inputs.index_name}} \
                --query_result_dir ${{inputs.query_result}} \
                --eval_result_dir ${{outputs.eval_result}} \
                --mlflow_tracking_uri ${{inputs.mlflow_tracking_uri}}"""
        + f" --keyvault {environment.azure_key_vault_endpoint}",
        environment=pipeline_job_env,
        environment_variables={
            "LOGGING_LEVEL": os.getenv("LOGGING_LEVEL", ""),
            "MAX_WORKER_THREADS": os.getenv("MAX_WORKER_THREADS", ""),
        },
    )

    job_name = f"{config.job_name}_{formatted_datetime_suffix()}"

    @dsl.pipeline(
        name=job_name,
        experiment_name=config.experiment_name,
        compute=environment.aml_compute_name,
        description=config.job_description or "RAG Experiment Pipeline",
        display_name=job_name,
    )
    def rag_pipeline(
        config_path_input,
        data_input,
        eval_data_input,
        mlflow_tracking_uri,
        mlflow_parent_run_id,
    ):
        index_job = index_pipeline_component(
            data=data_input,
            config_path=config_path_input,
            mlflow_tracking_uri=mlflow_tracking_uri,
        )

        query_job = query_pipeline_component(
            index_name=index_job.outputs.index_name,
            config_path=config_path_input,
            eval_data=eval_data_input,
            mlflow_tracking_uri=mlflow_tracking_uri,
        )

        eval_job = eval_pipeline_component(
            index_name=index_job.outputs.index_name,
            config_path=config_path_input,
            query_result=query_job.outputs.query_result,
            mlflow_tracking_uri=mlflow_tracking_uri,
        )

        return {"eval_result": eval_job.outputs.eval_result}

    # Save the environment into Keyvault for the pipeline steps to retrieve later
    environment.to_keyvault()

    pipeline = rag_pipeline(
        config_path_input=Input(type="uri_file", path=config_path),
        data_input=Input(type="uri_folder", path=config.data_dir),
        eval_data_input=Input(type="uri_file", path=config.eval_data_jsonl_file_path),
        mlflow_tracking_uri=mlflow_client.tracking_uri,
    )
    return ml_client.jobs.create_or_update(
        pipeline, experiment_name=config.experiment_name
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, help="path to the data folder", default="./data"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="relative path to the config file",
        default="./config.json",
    )
    args = parser.parse_args()

    environment = Environment.from_env_or_keyvault()
    config = Config(environment, args.config_path, args.data_dir)
    config.execution_environment = ExecutionEnvironment.AZURE_ML

    if config.sampling:
        logger.error(
            "Can't sample data when running on AzureML pipeline. Please run the pipeline locally"
        )
        exit()

    mlflow_client = initialise_mlflow_client(environment, config)

    # Starting multiple pipelines hence unable to stream them
    for index_config in config.index_configs():
        # with mlflow.start_run(run_name=config.job_name, description=config.job_description, experiment_id=experiment.experiment_id) as run:
        logger.info(f"Starting pipeline for index: {index_config.index_name()}")
        job = start_pipeline(
            environment, config, index_config, args.config_path, mlflow_client
        )
        logger.info(
            f"Pipeline job started...\nIndex name: {index_config.index_name()}\nMonitoring url: {job.studio_url}"
        )
