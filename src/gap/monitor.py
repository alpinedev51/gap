import torch

from gap.utils import logger


def print_gpu_memory_stats():
    """
    Prints a breakdown of the current GPU memory usage.
    """
    if not torch.cuda.is_available():
        logger.info("GPU not available. Running on CPU.")
        return

    gpu_name = torch.cuda.get_device_name(0)

    allocated = torch.cuda.memory_allocated(0) / 1024**3  # Convert bytes to GB

    reserved = torch.cuda.memory_reserved(0) / 1024**3

    total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    logger.info(f"--- GPU: {gpu_name} ---")
    logger.info(f" -- Allocated (Tensors): {allocated:.2f} GB")
    logger.info(f" -- Reserved (Cached):   {reserved:.2f} GB")
    logger.info(f" -- Total Capacity:      {total:.2f} GB")
    logger.info(f" -- Free (Approx):       {total - reserved:.2f} GB")
    logger.info("---------------------------------------")


def clear_gpu_cache():
    """
    Releases unused cached memory back to the OS.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU Cache Cleared.")
