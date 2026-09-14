"""Vision inference engine using Together AI's hosted API.

Handles image encoding, API calls, rate limiting, and JSON truncation recovery.
"""

import base64
import io
import json
import os
import re
import time
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv(override=True)

TOGETHER_API_URL = os.environ.get(
    "TOGETHER_API_URL", "https://api.together.xyz/v1/chat/completions"
)
TOGETHER_MODEL = os.environ.get("TOGETHER_MODEL", "Qwen/Qwen3.5-9B")
TOGETHER_MAX_TOKENS = int(os.environ.get("TOGETHER_MAX_TOKENS", "4096"))
TOGETHER_TEMPERATURE = float(os.environ.get("TOGETHER_TEMPERATURE", "0.0"))
TOGETHER_MAX_RETRIES = int(os.environ.get("TOGETHER_MAX_RETRIES", "15"))


class VisionEngine:
    """Wrapper for Together AI's hosted vision API.

    Handles authentication and model parameters without requiring local
    model downloads or GPU/CPU inference.
    """

    def __init__(self, backend: str, **parts):
        """
        Initialize the vision engine.

        Args:
            backend (str): Backend type (e.g., 'together')
            **parts: Additional configuration (e.g., api_key)
        """
        self.backend = backend
        self.parts = parts


def load_model() -> VisionEngine:
    """
    Load and initialize the vision model via Together AI's hosted API.

    Requires TOGETHER_API_KEY environment variable to be set.

    Returns:
        VisionEngine: Initialized vision engine with API credentials

    Raises:
        RuntimeError: If TOGETHER_API_KEY is not set
    """
    together_api_key = os.environ.get("TOGETHER_API_KEY")
    if not together_api_key:
        raise RuntimeError(
            "TOGETHER_API_KEY environment variable is not set. This pipeline runs "
            "inference through Together AI's hosted API only - set TOGETHER_API_KEY and retry."
        )
    print(f"✓ TOGETHER_API_KEY found - using Together AI's hosted '{TOGETHER_MODEL}' vision model.")
    return VisionEngine("together", api_key=together_api_key)


def _build_full_prompt(text_prompt: str, json_schema_str: Optional[str]) -> str:
    """
    Build comprehensive prompt with instructions and schema.

    Args:
        text_prompt (str): Domain-specific base prompt
        json_schema_str (str): JSON schema template

    Returns:
        str: Full prompt with detailed instructions
    """
    if not json_schema_str:
        return text_prompt

    return (
        f"{text_prompt}\n\n"
        f"Look carefully at the actual image provided and describe only what is "
        f"really visible in it. Do not reuse the sample id/type/material values "
        f"shown in the schema below - they are placeholders illustrating the "
        f"shape of the JSON, not the expected answer. If the image contains no "
        f"elements of a given kind, omit them or return an empty list.\n\n"
        f"Element ids must be the actual text label printed on the drawing next "
        f"to that element (e.g. 'B1', 'C12'). Never invent a sequential id "
        f"(1, 2, 3...) and never reuse the same id for more than one distinct "
        f"element - if you cannot clearly read a label, use null for that id "
        f"rather than guessing. If a page has many repeated elements of the "
        f"same type (more than ~12), do not list every single one - list up "
        f"to 10 real labels you can actually read, and note the approximate "
        f"total count and pattern in the insights field instead of enumerating "
        f"the rest.\n\n"
        f"For each element/component, beyond the fields shown in the schema, "
        f"add any other properties you can actually determine from the "
        f"drawing that would be useful - e.g. width, height, length, span, "
        f"quantity, grid_location, level, spacing, connection_type. Add them "
        f"as extra key-value pairs on that same object using short lowercase "
        f"snake_case keys. Only add a property when you can actually read or "
        f"clearly determine it from the drawing - never fabricate a value, "
        f"and don't force the same extra fields onto every element if they "
        f"don't genuinely apply to all of them.\n\n"
        f"Write the JSON keys in exactly the order they appear in the schema "
        f"below - summary, page_type, insights, confidence, confidence_notes, "
        f"then the elements/components list last. This matters because if "
        f"you run out of room, only the list at the end gets cut off; the "
        f"fields before it are what make the output useful even then.\n\n"
        f"Output JSON Schema (shape only, not real content):\n{json_schema_str}\n\n"
        f"Return JSON only inside ```json block. Keep element/component field "
        f"values concise (short labels/numbers), but write summary, insights, "
        f"and confidence_notes in full, detailed sentences as instructed above "
        f"- those are the fields meant to carry real depth, not just the "
        f"element list."
    )


def _extract_json_block(raw_response: str) -> str:
    """
    Extract JSON block from model response.

    Handles both complete ```json...``` fences and truncated responses.

    Args:
        raw_response (str): Raw model response text

    Returns:
        str: Extracted JSON text (may be incomplete if truncated)
    """
    json_match = re.search(r"```json\s*([\s\S]*?)\s*```", raw_response)
    if json_match:
        return json_match.group(1).strip()

    # No closing fence - response was likely truncated by max_tokens
    stripped = raw_response.strip()
    for opener in ("```json", "```"):
        if stripped.startswith(opener):
            return stripped[len(opener) :].lstrip()
    return stripped


def _repair_truncated_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort recovery for JSON truncated by token limit.

    Finds the last complete JSON object/array and auto-closes unclosed braces.

    Args:
        text (str): Potentially truncated JSON text

    Returns:
        dict: Parsed JSON object, or None if repair fails
    """
    text = text.strip()
    last_close = max(text.rfind("}"), text.rfind("]"))
    if last_close == -1:
        return None
    text = text[:last_close + 1]

    # Track open braces/brackets to auto-close
    closers = []
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            closers.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if closers:
                closers.pop()
    text += "".join(reversed(closers))

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _image_to_data_uri(image: Image.Image) -> str:
    """
    Convert PIL Image to base64 data URI.

    Args:
        image (Image.Image): PIL Image object

    Returns:
        str: Data URI string (data:image/png;base64,...)
    """
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _run_together(
    image: Image.Image,
    full_prompt: str,
    api_key: str,
    max_retries: int = TOGETHER_MAX_RETRIES,
    max_tokens: int = TOGETHER_MAX_TOKENS,
) -> Dict[str, Any]:
    """
    Call Together AI's hosted vision model.

    Handles rate limiting (429) with exponential backoff.

    Args:
        image (Image.Image): Input image
        full_prompt (str): Complete prompt with instructions
        api_key (str): Together AI API key
        max_retries (int): Max retry attempts for rate limits
        max_tokens (int): Max tokens in response

    Returns:
        dict: Response with keys: content, finish_reason, retries, elapsed_seconds

    Raises:
        RuntimeError: If rate limit exceeded after all retries
    """
    data_uri = _image_to_data_uri(image)
    payload = {
        "model": TOGETHER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": full_prompt},
                ],
            }
        ],
        "temperature": TOGETHER_TEMPERATURE,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    delay = 2.0
    start = time.monotonic()
    for attempt in range(max_retries):
        response = requests.post(
            TOGETHER_API_URL, json=payload, headers=headers, timeout=90
        )
        if response.status_code == 429:
            reset = response.headers.get(
                "x-ratelimit-reset", response.headers.get("Retry-After", delay)
            )
            time.sleep(min(float(reset), 65.0))
            delay = min(delay * 2, 65.0)
            continue
        response.raise_for_status()
        choice = response.json()["choices"][0]
        return {
            "content": choice["message"]["content"].strip(),
            "finish_reason": choice.get("finish_reason"),
            "retries": attempt,
            "elapsed_seconds": time.monotonic() - start,
        }

    raise RuntimeError(
        "Together AI rate limit exceeded after retries - try again shortly or reduce concurrent pages."
    )


def perform_fast_inference(
    image: Image.Image,
    text_prompt: str,
    json_schema_str: Optional[str] = None,
    engine: Optional[VisionEngine] = None,
) -> Dict[str, Any]:
    """
    Execute inference on an image via Together AI's hosted API.

    Args:
        image (Image.Image): Input image
        text_prompt (str): Domain-specific prompt
        json_schema_str (str): JSON schema template
        engine (VisionEngine): Initialized vision engine

    Returns:
        dict: Response with keys: content, json_str, finish_reason, retries, elapsed_seconds
    """
    full_prompt = _build_full_prompt(text_prompt, json_schema_str)
    result = _run_together(image, full_prompt, engine.parts["api_key"])
    result["json_str"] = _extract_json_block(result["content"])
    return result
