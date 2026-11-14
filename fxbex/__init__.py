"""Model package download and execution runtime."""
__version__ = "0.9.0"

from .llm import LLM
from .model import Model, YOLO, ObjectDetection, ImageGeneration
__all__ = ['Model', 'LLM', 'YOLO', 'ObjectDetection', 'ImageGeneration','hug','describe','example_inputs','as_torch','GraphFXB','ExecutionFXB','SafeTensorWeights']

from .graph_fxb import GraphFXB
from .execution_fxb import ExecutionFXB
from .checkpoint import SafeTensorWeights

from .schema import describe, example_inputs

from .hug import hug


def as_torch(package, *, execution_policy='npu_only', inference_only=True, output_device='cpu'):
    """Opt-in immutable nn.Module wrapper; importing FXBEX alone does not import torch."""
    from .torch_api import CompiledModule
    return CompiledModule(package,execution_policy=execution_policy,inference_only=inference_only,output_device=output_device)
