from . import datasets
from torch.utils.data import DataLoader, Subset
import numpy as np

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

# def prepare_dataloader(mode='training', dataset='Sintel', shuffle=False, batch_size=1, small_run=False, sintel_subsplit=False, dstype='clean', image_size=2, demo_path=None, n_images=-1):
#     """Get a PyTorch dataloader for the specified dataset

#     Args:
#             mode (str, optional):
#                     Specify the split of the dataset [training | evaluation]. Defaults to 'training'.
#             dataset (str, optional):
#                     Specify the dataset used [Sintel | Kitti15]. Defaults to 'Sintel'.
#             shuffle (bool, optional):
#                     Use random sampling. Defaults to False.
#             batch_size (int, optional):
#                     Defaults to 1.
#             small_run (bool, optional):
#                     For debugging: Will load only 32 images. Defaults to False.
#             sintel_subsplit (bool, optional): 
#                     Specific for Sintel dataset. If a subsplit is available (and the path correctly specified under config_paths.py),
#                     will use the subsplit for Sintel. Defaults to False.
#             dstype (str, optional):
#                     Specific for Sintel dataset. Dataset type [clean | final] . Defaults to 'clean'.

#     Raises:
#             ValueError: Unknown mode.
#             ValueError: Unkown dataset.

#     Returns:
#             torch.utils.data.DataLoader: Dataloader which can be used for FGSM.
#     """

#     if dataset == 'Sintel':
#         if not sintel_subsplit:
#             if mode == 'training':
#                 dataset = datasets.MpiSintel(split=Paths.splits("sintel_train"),
#                                              root=Paths.config("sintel_mpi"), dstype=dstype, has_gt=True, frames=image_size)
#             elif mode == 'evaluation':
#                 # with this option, ground truth and valid are None!!
#                 dataset = datasets.MpiSintel(split=Paths.splits("sintel_eval"),
#                                              root=Paths.config("sintel_mpi"), dstype=dstype, has_gt=False, frames=image_size)
#             else:
#                 raise ValueError(f'The specified mode: {mode} is unknown.')
#         else:
#             if mode == 'training':
#                 dataset = datasets.MpiSintelSubsplit(split=Paths.splits("sintel_sub_train"),
#                                                      root=Paths.config("sintel_subsplit"), dstype=dstype, has_gt=True, frames=image_size)
#             elif mode == 'evaluation':
#                 dataset = datasets.MpiSintelSubsplit(split=Paths.splits("sintel_sub_eval"),
#                                                      root=Paths.config("sintel_subsplit"), dstype=dstype, has_gt=True, frames=image_size)
#             else:
#                 raise ValueError(f'The specified mode: {mode} is unknown.')

#     elif dataset == 'Kitti15':
#         if mode == 'training':
#             dataset = datasets.KITTI(split=Paths.splits(
#                 "kitti_train"), aug_params=None, root=Paths.config("kitti15"), has_gt=True)
#         elif mode == 'evaluation':
#             dataset = datasets.KITTI(split=Paths.splits(
#                 "kitti_eval"), aug_params=None, root=Paths.config("kitti15"), has_gt=False)
#     elif dataset == 'Demo':
#         dataset = datasets.Demo(
#             root=demo_path, frames=image_size, n_images=n_images)
#     else:
#         raise ValueError(
#             "Unknown dataset %s, use either 'Sintel' or 'Kitti15'." % (dataset))

#     # if e.g. the evaluation dataset does not provide a ground truth this is specified
#     ds_has_gt = dataset.has_groundtruth()

#     if small_run:
#         rand_indices = np.random.randint(0, len(dataset), 32)
#         indices = np.arange(0, 32)
#         dataset = Subset(dataset, indices)

#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), ds_has_gt


def prepare_dataloader(mode='training', dataset_name='Sintel', shuffle=False, batch_size=1,
                       small_run=False, image_size=2, demo_path=None, n_images=-1, has_depth=True):
    """
    Get a PyTorch dataloader for the specified dataset using arguments from a configuration file.

    Args:
        mode (str): Split of the dataset ['training' | 'evaluation'].
        dataset_name (str): Name of the dataset ['Sintel' | 'Kitti15' | 'Demo'].
        shuffle (bool): Use random sampling. Defaults to False.
        batch_size (int): Batch size for the DataLoader. Defaults to 1.
        small_run (bool): For debugging: load only a subset of images. Defaults to False.
        image_size (int): Number of image frames per example. Defaults to 2.
        demo_path (str): Path to Demo dataset. Required if dataset_name is 'Demo'.
        n_images (int): Number of images to load for the Demo dataset. Defaults to -1 (all images).

    Returns:
        torch.utils.data.DataLoader: Dataloader for the specified dataset.
    """
    # Load dataset-specific configuration
    dataset_args = load_dataset_args(dataset_name)
    if not dataset_args:
        raise ValueError(
            f"Dataset configuration for {dataset_name} not found.")

    # Update dataset arguments with runtime parameters
    if dataset_name.lower() == 'kitti15' and mode.lower() == 'testing':
        dataset_args['split'] = 'testing'
        dataset_args['has_gt'] = False
        dataset_args['has_depth'] = False
    else:
        dataset_args['split'] = mode
    dataset_args['frames'] = image_size
    if dataset_name == 'Demo':
        dataset_args['root'] = demo_path
        dataset_args['n_images'] = n_images

    # Instantiate dataset
    if dataset_name == 'Sintel':
        dataset = datasets.MpiSintel(**dataset_args)
    elif dataset_name == 'Kitti15':
        dataset = datasets.KITTI(**dataset_args)
    elif dataset_name == 'Demo':
        dataset = datasets.Demo(**dataset_args)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}.")

    ds_has_gt = dataset.has_groundtruth()

    # Optionally limit dataset size for debugging
    if small_run:
        indices = np.random.choice(
            len(dataset), min(32, len(dataset)), replace=False)
        dataset = Subset(dataset, indices)

    # Create DataLoader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, ds_has_gt
