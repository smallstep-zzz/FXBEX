"""Independent FXBEX package builder and installed-adapter runtime."""
__version__ = "0.4.2"

from .llm import LLM
from .model import Model, YOLO, ObjectDetection, ImageGeneration
__all__ = ['Model', 'LLM', 'YOLO', 'ObjectDetection', 'ImageGeneration']

from .schema import describe, example_inputs
