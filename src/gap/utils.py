import logging
import os

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
    return torch.device("cpu")
