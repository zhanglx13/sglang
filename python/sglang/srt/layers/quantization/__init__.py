# temporarily adapted from vLLM
# FIXME: in progress of refactoring the model loader

from typing import Any, Dict, List, Optional, Type

import torch
from torch.nn.parameter import Parameter
from vllm.model_executor.layers.quantization.aqlm import AQLMConfig
from vllm.model_executor.layers.quantization.awq import AWQConfig
from vllm.model_executor.layers.quantization.awq_marlin import AWQMarlinConfig
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.quantization.bitsandbytes import BitsAndBytesConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig,
)
from vllm.model_executor.layers.quantization.deepspeedfp import DeepSpeedFPConfig
from vllm.model_executor.layers.quantization.fbgemm_fp8 import (
    FBGEMMFp8Config,
    FBGEMMFp8LinearMethod,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8LinearMethod
from vllm.model_executor.layers.quantization.gptq import GPTQConfig
from vllm.model_executor.layers.quantization.gptq_marlin import GPTQMarlinConfig
from vllm.model_executor.layers.quantization.gptq_marlin_24 import GPTQMarlin24Config
from vllm.model_executor.layers.quantization.marlin import MarlinConfig
from vllm.model_executor.layers.quantization.squeezellm import SqueezeLLMConfig
from vllm.model_executor.utils import set_weight_attrs


def fp8_linear_create_weights(
    self,
    layer: torch.nn.Module,
    input_size_per_partition: int,
    output_partition_sizes: List[int],
    input_size: int,
    output_size: int,
    params_dtype: torch.dtype,
    **extra_weight_attrs,
):
    del input_size, output_size
    output_size_per_partition = sum(output_partition_sizes)

    layer.logical_widths = output_partition_sizes

    layer.input_size_per_partition = input_size_per_partition
    layer.output_size_per_partition = output_size_per_partition
    layer.orig_dtype = params_dtype

    # WEIGHT
    # weight_dtype = (torch.float8_e4m3fn
    #                 if self.quant_config.is_checkpoint_fp8_serialized else
    #                 params_dtype)
    weight_dtype = torch.float8_e4m3fn
    weight = Parameter(
        torch.empty(
            output_size_per_partition, input_size_per_partition, dtype=weight_dtype
        ),
        requires_grad=False,
    )
    layer.register_parameter("weight", weight)
    set_weight_attrs(
        weight,
        {
            **extra_weight_attrs,
            "input_dim": 1,
            "output_dim": 0,
        },
    )

    # If checkpoint is serialized fp8, load them.
    # Otherwise, wait until process_weights_after_loading.
    if self.quant_config.is_checkpoint_fp8_serialized:
        # WEIGHT SCALE
        scale = create_per_tensor_scale_param(
            output_partition_sizes, **extra_weight_attrs
        )
        layer.register_parameter("weight_scale", scale)

        # INPUT ACTIVATION SCALE
        if self.quant_config.activation_scheme == "static":
            scale = create_per_tensor_scale_param(
                output_partition_sizes, **extra_weight_attrs
            )
            layer.register_parameter("input_scale", scale)


setattr(Fp8LinearMethod, "create_weights", fp8_linear_create_weights)

QUANTIZATION_METHODS: Dict[str, Type[QuantizationConfig]] = {
    "aqlm": AQLMConfig,
    "awq": AWQConfig,
    "deepspeedfp": DeepSpeedFPConfig,
    "fp8": Fp8Config,
    "fbgemm_fp8": FBGEMMFp8Config,
    # The order of gptq methods is important for config.py iteration over
    # override_quantization_method(..)
    "marlin": MarlinConfig,
    "gptq_marlin_24": GPTQMarlin24Config,
    "gptq_marlin": GPTQMarlinConfig,
    "awq_marlin": AWQMarlinConfig,
    "gptq": GPTQConfig,
    "squeezellm": SqueezeLLMConfig,
    "compressed-tensors": CompressedTensorsConfig,
    "bitsandbytes": BitsAndBytesConfig,
}


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")
    return QUANTIZATION_METHODS[quantization]


__all__ = [
    "QuantizationConfig",
    "get_quantization_config",
    "QUANTIZATION_METHODS",
]
