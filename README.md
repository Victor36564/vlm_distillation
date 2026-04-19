# VLM Distillation (LLaVA)

Small toolkit for training and serving a custom vision-language model (VLM) using a vision encoder + LoRA-tuned language model + projector.

## Main Files

- `vlm_distill_LLaVA.py`: Train pipeline for LLaVA-style data (`llava_images_100k/`). Builds model, trains, and saves checkpoints.
- `test_LLaVA.py`: Loads a trained checkpoint and runs single-sample inference on the dataset split.
- `run_model_LLaVA.py`: FastAPI server for inference (`/chat`) from local image path or base64 image.
- `data_extract.py`: Dataset/data extraction helper.
- `requirements.txt`: Python dependencies.
- `checkpoints/`: Saved LoRA adapters + projector weights.
- `images_test/`: Local images for quick inference testing.

## Quick Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train (LLaVA)

```bash
python vlm_distill_LLaVA.py
```

Outputs are saved under `<vision>__<lm>__LLaVA/`.

## Test a Trained Model

```bash
python test_LLaVA.py
```

## Run API Server

```bash
python run_model_LLaVA.py
```

Server starts on `http://0.0.0.0:8000` with endpoint:

- `POST /chat`
  - fields: `prompt`, and either `image_path` or `image_base64`

Example request body:

```json
{
  "prompt": "Summarize the image in one sentence.",
  "image_path": "images_test/bowl.jpg"
}
```
