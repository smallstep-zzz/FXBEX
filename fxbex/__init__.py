"""Independent FXBEX package builder and installed-adapter runtime."""
__version__ = "0.6.4"

from .llm import LLM
from .model import Model, YOLO, ObjectDetection, ImageGeneration
__all__ = ['Model', 'LLM', 'YOLO', 'ObjectDetection', 'ImageGeneration','hug','describe','example_inputs']

from .schema import describe, example_inputs

from .hug import hug
