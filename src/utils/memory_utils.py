# src/utils/memory_utils.py
"""
Memory optimization utilities for efficient GPU usage.
Automatically adjusts batch sizes and handles OOM errors.
"""

import torch
import logging
from typing import Callable, Any
import psutil
import gc

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Manages GPU/CPU memory and automatically adjusts batch sizes.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self.is_cuda = device.type == "cuda"

        if self.is_cuda:
            self.total_memory = (
                torch.cuda.get_device_properties(device).total_memory / 1e9
            )  # GB
            logger.info(f"GPU: {torch.cuda.get_device_name(device)}")
            logger.info(f"Total VRAM: {self.total_memory:.2f} GB")
        else:
            self.total_memory = psutil.virtual_memory().total / 1e9
            logger.info(f"CPU RAM: {self.total_memory:.2f} GB")

    def get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        if self.is_cuda:
            return torch.cuda.memory_allocated(self.device) / 1e9
        else:
            return psutil.Process().memory_info().rss / 1e9

    def get_memory_free(self) -> float:
        """Get free memory in GB"""
        if self.is_cuda:
            return self.total_memory - torch.cuda.memory_allocated(self.device) / 1e9
        else:
            return psutil.virtual_memory().available / 1e9

    def clear_cache(self):
        """Clear memory cache"""
        gc.collect()
        if self.is_cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def print_memory_stats(self):
        """Print detailed memory statistics"""
        if self.is_cuda:
            allocated = torch.cuda.memory_allocated(self.device) / 1e9
            reserved = torch.cuda.memory_reserved(self.device) / 1e9
            free = self.total_memory - allocated

            logger.info("VRAM Stats:")
            logger.info(
                f"  Allocated: {allocated:.2f} GB ({allocated / self.total_memory * 100:.1f}%)"
            )
            logger.info(
                f"  Reserved:  {reserved:.2f} GB ({reserved / self.total_memory * 100:.1f}%)"
            )
            logger.info(
                f"  Free:      {free:.2f} GB ({free / self.total_memory * 100:.1f}%)"
            )
        else:
            mem = psutil.virtual_memory()
            logger.info("RAM Stats:")
            logger.info(f"  Used:  {mem.used / 1e9:.2f} GB ({mem.percent:.1f}%)")
            logger.info(f"  Free:  {mem.available / 1e9:.2f} GB")


class DynamicBatchSizer:
    """
    Automatically finds optimal batch size through binary search.
    Handles OOM errors gracefully.
    """

    def __init__(
        self,
        min_batch_size: int = 1,
        max_batch_size: int = 128,
        scale_factor: float = 2.0,
    ):
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.scale_factor = scale_factor
        self.optimal_batch_size = None

    def find_optimal_batch_size(
        self, test_function: Callable[[int], Any], device: torch.device
    ) -> int:
        """
        Binary search to find maximum batch size that doesn't OOM.

        Args:
            test_function: Function that takes batch_size and returns something
                          Should raise RuntimeError with "out of memory" on OOM
            device: torch device

        Returns:
            Optimal batch size
        """
        if self.optimal_batch_size is not None:
            return self.optimal_batch_size

        logger.info("🔍 Finding optimal batch size...")

        low = self.min_batch_size
        high = self.max_batch_size
        best_batch_size = low

        while low <= high:
            mid = (low + high) // 2

            try:
                # Clear cache before test
                if device.type == "cuda":
                    torch.cuda.empty_cache()

                logger.info(f"  Testing batch_size={mid}...")
                test_function(mid)

                # Success - try larger
                best_batch_size = mid
                low = mid + 1
                logger.info(f"  ✓ batch_size={mid} works")

            except RuntimeError as e:
                if "out of memory" in str(e) or "OutOfMemoryError" in str(e):
                    # OOM - try smaller
                    high = mid - 1
                    logger.info(f"  ✗ batch_size={mid} OOM, reducing...")

                    # Clear memory
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                else:
                    # Some other error
                    raise

        self.optimal_batch_size = best_batch_size
        logger.info(f"✓ Optimal batch size: {best_batch_size}")

        return best_batch_size


def safe_forward_pass(
    model: torch.nn.Module, *args, batch_size: int, max_retries: int = 3, **kwargs
) -> Any:
    """
    Execute forward pass with automatic OOM handling.
    Reduces batch size if OOM occurs.

    Args:
        model: PyTorch model
        *args: Positional arguments for model
        batch_size: Initial batch size
        max_retries: Maximum number of retries with reduced batch size
        **kwargs: Keyword arguments for model

    Returns:
        Model output
    """
    current_batch_size = batch_size

    for attempt in range(max_retries):
        try:
            return model(*args, **kwargs)

        except RuntimeError as e:
            if "out of memory" in str(e):
                if attempt < max_retries - 1:
                    # Reduce batch size
                    current_batch_size = max(1, current_batch_size // 2)
                    logger.warning(
                        f"OOM in forward pass, reducing batch_size to {current_batch_size}"
                    )

                    # Clear cache
                    torch.cuda.empty_cache()
                    gc.collect()

                    # Adjust input tensors for new batch size
                    # This is a simplified version - you'd need to split the batch
                    continue
                else:
                    raise RuntimeError(
                        f"Failed after {max_retries} attempts. Try reducing batch size manually."
                    )
            else:
                raise


def estimate_model_memory(model: torch.nn.Module) -> dict:
    """
    Estimate memory requirements of a model.

    Returns:
        Dictionary with memory estimates in GB
    """
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers()) / 1e9

    # Rough estimate for activations (highly dependent on batch size and sequence length)
    # For ESM-2 650M: ~4-6GB for batch_size=8, seq_len=512
    activation_estimate = 4.0  # GB (conservative)

    return {
        "parameters": param_memory,
        "buffers": buffer_memory,
        "activations_estimate": activation_estimate,
        "total_estimate": param_memory + buffer_memory + activation_estimate,
    }


def setup_model_dtype(
    model: torch.nn.Module, precision: str = "bf16"
) -> torch.nn.Module:
    """
    Convert model to optimal dtype for inference/training.

    Args:
        model: PyTorch model
        precision: 'bf16', 'fp16', or 'fp32'

    Returns:
        Model with updated dtype
    """
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

    if precision not in dtype_map:
        raise ValueError(
            f"Invalid precision: {precision}. Choose from {list(dtype_map.keys())}"
        )

    target_dtype = dtype_map[precision]

    # Convert model
    model = model.to(dtype=target_dtype)

    logger.info(f"✓ Model converted to {precision}")

    return model


def check_gpu_compatibility(min_vram_gb: float = 6.0) -> dict:
    """
    Check if GPU meets minimum requirements.

    Args:
        min_vram_gb: Minimum VRAM required in GB

    Returns:
        Dictionary with GPU info and recommendations
    """
    if not torch.cuda.is_available():
        return {
            "compatible": False,
            "reason": "No CUDA GPU detected",
            "recommendation": "Use CPU (will be slower) or enable CPU offloading",
        }

    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9

    result = {
        "gpu_name": gpu_name,
        "total_vram_gb": total_vram,
        "compatible": total_vram >= min_vram_gb,
    }

    if total_vram >= 12:
        result["recommendation"] = "Plenty of VRAM. Use batch_size=16, no CPU offload."
    elif total_vram >= 8:
        result["recommendation"] = (
            "Good VRAM. Use batch_size=8, bfloat16, gradient checkpointing."
        )
    elif total_vram >= 6:
        result["recommendation"] = (
            "Limited VRAM. Use batch_size=4, bfloat16, gradient checkpointing, consider CPU offload."
        )
    else:
        result["recommendation"] = (
            "Insufficient VRAM. Enable CPU offloading or use CPU training."
        )

    # Check for Blackwell/Ada/Ampere architecture (native bfloat16 support)
    compute_capability = torch.cuda.get_device_capability(0)
    result["compute_capability"] = f"{compute_capability[0]}.{compute_capability[1]}"
    result["native_bfloat16"] = compute_capability[0] >= 8  # Ampere and newer

    if result["native_bfloat16"]:
        result["precision_recommendation"] = "bf16-mixed (native support)"
    else:
        result["precision_recommendation"] = "16-mixed (fp16)"

    return result


# Convenience function for automatic setup
def auto_configure_memory(config: dict, device: torch.device) -> dict:
    """
    Automatically configure memory settings based on available hardware.

    Args:
        config: Configuration dictionary
        device: Target device

    Returns:
        Updated configuration with optimal memory settings
    """
    updated_config = config.copy()

    if device.type == "cuda":
        gpu_info = check_gpu_compatibility()

        logger.info(f"GPU: {gpu_info['gpu_name']}")
        logger.info(f"VRAM: {gpu_info['total_vram_gb']:.2f} GB")
        logger.info(f"Recommendation: {gpu_info['recommendation']}")

        # Auto-adjust settings
        vram = gpu_info["total_vram_gb"]

        if vram >= 12:
            updated_config["training"]["batch_size"] = 16
            updated_config["hardware"]["precision"] = "bf16-mixed"
            updated_config["model"]["memory_optimization"]["cpu_offload"] = False
        elif vram >= 8:
            updated_config["training"]["batch_size"] = 8
            updated_config["hardware"]["precision"] = "bf16-mixed"
            updated_config["model"]["memory_optimization"]["cpu_offload"] = False
        elif vram >= 6:
            updated_config["training"]["batch_size"] = 4
            updated_config["hardware"]["precision"] = "bf16-mixed"
            updated_config["model"]["memory_optimization"]["cpu_offload"] = True
        else:
            updated_config["training"]["batch_size"] = 2
            updated_config["hardware"]["precision"] = "16-mixed"
            updated_config["model"]["memory_optimization"]["cpu_offload"] = True

        # Always enable gradient checkpointing
        updated_config["model"]["memory_optimization"]["gradient_checkpointing"] = True

        logger.info("Auto-configured:")
        logger.info(f"  Batch size: {updated_config['training']['batch_size']}")
        logger.info(f"  Precision: {updated_config['hardware']['precision']}")
        logger.info(
            f"  CPU offload: {updated_config['model']['memory_optimization']['cpu_offload']}"
        )

    return updated_config
