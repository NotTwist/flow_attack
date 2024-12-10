import yaml
import os

def get_config_path(config_name="datasets.yaml"):
    """Returns the absolute path to the configuration file located in the configs folder."""
    # Get the root directory of the project (one level up from 'utils')
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Build the path to the config folder
    config_dir = os.path.join(project_root, "configs")

    # Build the full path to the config file
    config_path = os.path.join(config_dir, config_name)

    # Check if the file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")

    return config_path


def load_config(config_path):
    """
    Load a configuration file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
        
    Returns:
        dict: Parsed configuration as a Python dictionary.
    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def load_dataset_args(dataset_name):
    """Load dataset arguments based on the dataset

    Args:
        dataset_name (string): Name of dataset
    """
    config = load_config(get_config_path("datasets.yaml"))
    return config['datasets'].get(dataset_name)