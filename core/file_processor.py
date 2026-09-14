"""File processing engine for extracting and downsampling images from PDFs and image files."""

import io
from typing import List

import fitz  # PyMuPDF
from PIL import Image


def process_uploaded_file_fast(
    file_name: str, file_bytes: bytes, dpi: int = 100, max_dimension: int = 768
) -> List[Image.Image]:
    """
    Extract and downsample images from PDF or image files.

    Supports PDF, PNG, JPG, JPEG, WEBP, BMP, TIFF formats.
    Images are downsampled to a maximum of 768px for rapid vision inference.

    Args:
        file_name (str): Filename with extension (determines format)
        file_bytes (bytes): Raw file bytes
        dpi (int): PDF render DPI (default 100)
        max_dimension (int): Maximum width/height in pixels (default 768)

    Returns:
        List[Image.Image]: List of PIL Image objects in RGB format

    Raises:
        ValueError: If file extension is not supported
        Exception: If file processing fails (logged and returns empty list)
    """
    ext = file_name.split(".")[-1].lower()
    extracted_images = []

    try:
        # PDF Processing
        if ext == "pdf":
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                zoom = dpi / 72
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))

                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                if max(img.size) > max_dimension:
                    img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                extracted_images.append(img)
            doc.close()

        # Direct Image Processing
        elif ext in ["jpg", "jpeg", "png", "webp", "bmp", "tiff"]:
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            if max(img.size) > max_dimension:
                img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            extracted_images.append(img)

        else:
            raise ValueError(f"Unsupported file extension: .{ext}")

        return extracted_images

    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
        return []
