import argparse

from rag_experiment_accelerator.run.qa_generation import run
from rag_experiment_accelerator.config.config import Config
from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.config.paths import get_all_file_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, help="input: path to the config file"
    )
    parser.add_argument("--data_dir", type=str, help="input: path to the input data")
    args, _ = parser.parse_known_args()

    environment = Environment.from_env_or_keyvault()
    print(environment.openai_api_version)
    config = Config(environment, args.config_path, args.data_dir)
    print(config.AZURE_OAI_CHAT_DEPLOYMENT_NAME)

    run(environment, config, get_all_file_paths(config.data_dir))
