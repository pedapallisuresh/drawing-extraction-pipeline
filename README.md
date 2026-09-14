# Drawing Extraction Pipeline

AI-powered structured data extraction from engineering drawings (PDF, images) using vision models. Extract components, elements, and metadata from Civil, Electrical, and Chip domain drawings in seconds.

## Features

✨ **Multi-Format Support**: PDF, PNG, JPG, JPEG, WEBP, BMP, TIFF
🎯 **Domain-Specific Extraction**: 
  - Civil Engineering (beams, columns, slabs, walls, footings)
  - Electrical Schematics (resistors, capacitors, ICs, relays)
  - Chip/Microelectronics (NAND, transistors, cells, blocks)

⚡ **Fast Processing**: Image downsampling, parallel inference (2 concurrent pages)
🔄 **Intelligent Truncation Recovery**: Auto-repairs JSON cut off by token limits
📊 **Interactive Dashboard**: Gradio UI with live stats, filtering, CSV export
🎨 **Rich Metadata**: Confidence ratings, insights, and domain-specific properties

## Prerequisites

- Python 3.8+
- Together AI API key ([Get one free](https://www.together.ai/))

## Installation

```bash
# Clone the repository
git clone https://github.com/pedapallisuresh/drawing-extraction-pipeline.git
cd drawing-extraction-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env and add your TOGETHER_API_KEY
```

## Quick Start

### Command Line

```bash
python main.py --file path/to/drawing.pdf --domain civil --max-pages 5
```

### Gradio Dashboard (Interactive)

```bash
python main.py --ui
```

Then open the browser URL displayed in the terminal.

## Project Structure

```
drawing-extraction-pipeline/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variables template
├── .gitignore                     # Git ignore rules
│
├── main.py                        # Entry point: CLI + Gradio launcher
│
├── core/
│   ├── __init__.py
│   ├── file_processor.py          # Image extraction & downsampling
│   ├── inference_engine.py        # Vision model API calls & truncation repair
│   └── schemas.py                 # Domain-specific schemas & prompts
│
├── ui/
│   ├── __init__.py
│   ├── gradio_app.py              # Gradio dashboard & components
│   ├── callbacks.py               # Extraction pipeline & post-processing
│   └── styles.py                  # CSS theming & styling
│
└── utils/
    ├── __init__.py
    ├── formatting.py              # DataFrame & CSV helpers
    └── status_rendering.py        # Live stats & status displays
```

## Usage

### 1. Single File Processing (CLI)

```python
from core.file_processor import process_uploaded_file_fast
from core.inference_engine import load_model, perform_fast_inference
from core.schemas import civil_prompt, civil_json_schema

# Load model once
engine = load_model()

# Process file
with open("drawing.pdf", "rb") as f:
    images = process_uploaded_file_fast("drawing.pdf", f.read())

# Extract from first page
result = perform_fast_inference(images[0], civil_prompt, civil_json_schema, engine=engine)
print(result['json_str'])
```

### 2. Interactive Dashboard

```bash
python main.py --ui
```

Upload a drawing, select domain, configure max pages, and click "Run Extraction". 
Results stream live with:
- Elements table (exportable as CSV)
- Page summaries with confidence
- Page previews
- Raw JSON for debugging

## API Reference

### `core.file_processor.process_uploaded_file_fast()`

```python
def process_uploaded_file_fast(file_name: str, file_bytes: bytes, dpi: int = 100, max_dimension: int = 768) -> List[Image.Image]:
    """
    Extract and downsample images from PDF or image files.
    
    Args:
        file_name: Filename (extension determines format)
        file_bytes: Raw file bytes
        dpi: PDF render DPI (default 100)
        max_dimension: Max width/height in pixels (default 768)
    
    Returns:
        List of PIL Image objects (RGB)
    """
```

### `core.inference_engine.perform_fast_inference()`

```python
def perform_fast_inference(image: Image.Image, text_prompt: str, json_schema_str: str, engine: VisionEngine) -> dict:
    """
    Run inference on an image via Together AI API.
    
    Args:
        image: PIL Image to analyze
        text_prompt: Instructions for the model
        json_schema_str: JSON schema template (shape + instructions)
        engine: VisionEngine with API credentials
    
    Returns:
        {
            'content': str,              # Raw model response
            'json_str': str,             # Extracted JSON block
            'finish_reason': str,        # 'stop' or 'length' (truncated)
            'retries': int,              # Number of rate-limit retries
            'elapsed_seconds': float     # API call duration
        }
    """
```

### `core.schemas`

```python
from core.schemas import civil_prompt, civil_json_schema
from core.schemas import electrical_prompt, electrical_json_schema
from core.schemas import chip_prompt, chip_json_schema
```

Each provides a domain-specific prompt and JSON schema template.

## Configuration

### Environment Variables (`.env`)

```env
TOGETHER_API_KEY=your_api_key_here
TOGETHER_MODEL=Qwen/Qwen3.5-9B
TOGETHER_API_URL=https://api.together.xyz/v1/chat/completions
```

### Inference Parameters

Edit `core/inference_engine.py`:
- `max_tokens`: Default 4096 (increase for complex drawings)
- `temperature`: Default 0.0 (deterministic)
- `max_retries`: Default 15 (rate limit retries)
- `max_workers`: Default 2 (parallel pages)

## JSON Output Schema

### Page Metadata

```json
{
  "summary": "2-4 sentence description of this drawing",
  "page_type": "Site Plan|Floor Plan|Detail|Elevation|Section|Mechanical Layout|...",
  "insights": [
    "Specific observation 1",
    "Specific observation 2",
    "Approximate count and pattern if elements are too numerous"
  ],
  "confidence": "high|medium|low",
  "confidence_notes": "Explanation of confidence rating with specifics",
  "elements": [
    {
      "id": "B1",
      "type": "Beam",
      "material": "Steel",
      "width": "12 in",
      "height": "18 in"
    }
  ]
}
```

Elements/components vary by domain but always include:
- `id`: Label from drawing (or null if unreadable)
- `type`: Component category
- `value`, `material`, or `location`: Domain-specific
- Extra properties: Any readable dimensions, quantities, or layout info

## Troubleshooting

### "TOGETHER_API_KEY is not set"
- Ensure `.env` file exists with valid API key
- Run: `export TOGETHER_API_KEY="your_key"` (Linux/Mac) or set it in Windows env

### Rate Limited (429 errors)
- Pipeline auto-retries up to 15 times with exponential backoff
- Reduce `max_pages` or `max_workers` if consistently hitting limits

### Truncated JSON (token limit)
- Pipeline auto-repairs truncated JSON
- For very complex drawings, increase `max_tokens` in `core/inference_engine.py`

### Poor Extraction Quality
- Ensure image clarity and contrast
- Verify domain matches drawing type
- Check `confidence` field in output for model uncertainty

## Examples

See `examples/` directory for sample inputs and outputs:
- `civil_floor_plan.pdf` → extracted elements
- `electrical_schematic.png` → circuit components
- `chip_layout.jpg` → silicon blocks

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Style

```bash
black . --line-length=100
flake8 . --max-line-length=100
```

### Adding New Domains

1. Define schema in `core/schemas.py`
2. Add domain option to `ui/gradio_app.py`
3. Test with sample drawings

## Performance

| Aspect | Metric |
|--------|--------|
| Single page | ~2-5 seconds (including API latency) |
| Parallel (2 pages) | ~3-7 seconds total |
| Image downsampling | ~0.1s per page |
| Max file size | ~50 MB (API limit) |
| Max pages | 200 (configurable) |

## Limitations

- Requires Internet connection (Together AI API)
- Drawing clarity affects extraction accuracy
- Very dense drawings may be truncated (token limit)
- Model may hallucinate labels if drawing is unclear

## License

MIT License — see LICENSE file

## Support

- Issues: [GitHub Issues](https://github.com/pedapallisuresh/drawing-extraction-pipeline/issues)
- Discussions: [GitHub Discussions](https://github.com/pedapallisuresh/drawing-extraction-pipeline/discussions)

## Citation

If you use this project in research, please cite:

```bibtex
@software{drawing_extraction_2024,
  title={Drawing Extraction Pipeline},
  author={Peda Pallisuru},
  url={https://github.com/pedapallisuresh/drawing-extraction-pipeline},
  year={2024}
}
```

---

**Built with ❤️ using Together AI Vision Models & Gradio**
