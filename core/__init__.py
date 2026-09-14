"""Core module for drawing extraction pipeline.

Provides file processing, vision inference, and domain-specific schemas.
"""

from .file_processor import process_uploaded_file_fast
from .inference_engine import load_model, perform_fast_inference, VisionEngine
from .schemas import (
    civil_prompt,
    civil_json_schema,
    electrical_prompt,
    electrical_json_schema,
    chip_prompt,
    chip_json_schema,
)

__all__ = [
    "process_uploaded_file_fast",
    "load_model",
    "perform_fast_inference",
    "VisionEngine",
    "civil_prompt",
    "civil_json_schema",
    "electrical_prompt",
    "electrical_json_schema",
    "chip_prompt",
    "chip_json_schema",
]
