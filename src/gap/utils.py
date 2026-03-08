import json
import logging
import os
import random
from datetime import datetime

import joblib
import numpy as np
import torch

LOGGER_FORMAT = "%(asctime)s %(name)16.16s %(levelname)s: %(message)s"


def get_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(level=log_level, format=LOGGER_FORMAT)
    logger = logging.getLogger(name)
    return logger


logger = get_logger(__name__)


class MissingEnvVar(Exception):
    def __init__(self, var_name, message="Required environment variable is missing"):
        self.var_name = var_name
        super().__init__(f"{message}: {var_name}")


def get_env_var(var_name: str) -> str:
    var = os.getenv(var_name)
    if var is None:
        error = MissingEnvVar(var_name)
        logger.error(error)
        raise error
    return var


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_reproducibility_seeds(seed=42):
    """Locks down all random number generators for exact reproducibility."""
    # 1. Python & NumPy
    random.seed(seed)
    np.random.seed(seed)

    # 2. PyTorch Base
    torch.manual_seed(seed)

    # 3. PyTorch Apple Silicon (MPS)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    # 4. PyTorch CUDA (Just in case you run this on a non-Mac later)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"All random seeds locked to {seed}.")


def save_experiment(model, model_name, config, class_names=None):
    """
    Creates a dedicated folder for the model run and saves:
    1. The model weights (.pth or .pkl)
    2. The configuration and metadata (.json)
    """
    # Create the root artifacts directory
    base_dir = "../saved_models"
    os.makedirs(base_dir, exist_ok=True)

    # Create a unique, descriptive folder for this specific run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = os.path.join(base_dir, f"{model_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # 1. Save Metadata (The Configuration)
    config_path = os.path.join(run_dir, "metadata.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    if class_names is not None:
        config['classes'] = list(class_names)

    # 2. Save Model Weights (Checks if it's PyTorch or Scikit-Learn)
    if isinstance(model, torch.nn.Module):
        # Save PyTorch state_dict (best practice)
        weights_path = os.path.join(run_dir, "weights.pth")
        torch.save(model.state_dict(), weights_path)
    else:
        # Save Scikit-Learn model via joblib
        weights_path = os.path.join(run_dir, "model.pkl")
        joblib.dump(model, weights_path)

    print(f"Model and metadata saved to: {run_dir}/")
    return run_dir
