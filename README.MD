# About

Video Cleaner is an application for detecting and removing anomalies, watermarks, and unwanted content from images and videos. It includes support for multiple detectors, inpainting backends, video trackers, and both terminal and visual user interfaces

## Features

- Image and video anomaly cleaning
- Multiple detection pipelines using OpenCV, YOLO, Grounding DINO, SAM, FFT, OCR, and anomaly detection
- Inpainting support via OpenCV, LaMa, Stable Diffusion XL (SDXL), and Flux
- Video tracking strategies: optical flow, Kalman filter, XMem, and CoTracker
- Preprocessing, mask refinement, inpainting, postprocessing, and audio passthrough
- CLI-driven operation and optional Gradio-based visual UI

## Installation

1. Create and activate a Python virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install the required packages.

```powershell
pip install -r requirements.txt
```

3. Optional: install additional model packages as needed.

```powershell
pip install torch torchvision timm diffusers transformers accelerate
pip install simple-lama-inpainting
pip install paddleocr easyocr ultralytics
```

## Usage

Run the application from the repository root:

```powershell
python main.py
```

This launches a startup launcher where you can choose between:

- `visual` — open the Gradio web interface(default)
- `terminal` — run the interactive terminal UI
- `run` — execute the pipeline directly with CLI options

### Run directly

```powershell
python main.py --program run --input path\to\input.mp4 --output path\to\output.mp4
```

### Common options

- `--config <path>`: load configuration from JSON
- `--input <path>`: input image or video path
- `--output <path>`: output image or video path
- `--type <auto|image|video>`: force input mode
- `--preset <default|fast|balanced|quality>`: pipeline preset
- `--inpainter <auto|opencv|lama|sdxl|flux>`: select inpainter backend
- `--tracker <none|optical_flow|kalman|xmem|cotracker>`: select tracker
- `--ocr <none|paddle|easy|both>`: OCR mode
- `--device <auto|cpu|cuda|mps>`: preferred device
- `--audio-backend <none|moviepy|ffmpeg>`: audio passthrough backend
- `--yes-install`: auto-install missing packages
- `--no-package-check`: skip package verification
- `--dry-run`: print resolved config without running
- `--print-config`: display the resolved configuration

### Detection and pipeline tuning

- `--detection-sensitivity <0-100>`: detection sensitivity
- `--mask-expand <int>`: expand detected mask area
- `--min-area <int>`: remove small mask components
- `--fusion-strictness <0-100>`: detector fusion strictness
- `--no-preprocessing`: disable preprocessing
- `--no-detection`: disable detection
- `--no-tracking`: disable video tracking
- `--no-inpainting`: disable inpainting
- `--no-postprocessing`: disable postprocessing
- `--no-audio`: disable audio copy
- `--save-intermediate`: save intermediate masks and frames

## Configuration

The application supports environment variables via a `.env` file and JSON config files. Environment variables use the `WATERMWARK_` prefix.

Example `.env` values:

```env
WATERMWARK_PROGRAM=auto
WATERMWARK_INPUT=inputs/example.mp4
WATERMWARK_OUTPUT=outputs/cleaned.mp4
WATERMWARK_INPAINTER=auto
WATERMWARK_TRACKER=optical_flow
WATERMWARK_OCR=none
```

## Project Structure

- `controls/` — command-line argument handling and pipeline control logic
- `core/` — configuration, logging, device/runtime utilities
- `detectors/` — image and video detection models
- `inpainters/` — inpainting implementations
- `mask_processing/` — mask refining and cleanup
- `pipelines/` — image/video processing pipelines
- `postprocessing/` — final enhancement and smoothing
- `preprocessing/` — image/video preprocessing filters
- `trackers/` — video tracking algorithms
- `ui/` — terminal and visual interfaces
- `utils/` — supporting utilities

## Notes

- The app is designed for anomaly and watermark cleaning, with a strong focus on configurable pipelines.
- For large models and GPU acceleration, ensure `torch` is installed and the correct device is selected.
- If using video input, choose `ffmpeg` or `moviepy` for audio handling.

## License


