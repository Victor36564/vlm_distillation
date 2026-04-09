# VLM Distillation

Train a simple vision-language captioning model by combining:

- A frozen SigLIP vision encoder: `google/siglip-base-patch16-256`
- A Qwen language model with LoRA adapters: `Qwen/Qwen2.5-0.5B-Instruct`
- A learned projector that maps vision features into language embedding space

The training script fine-tunes LoRA + projector for image caption generation and saves reusable checkpoint artifacts.

## Repository Structure

- `vlm_distill.py`: training pipeline, model definition, checkpoint saving
- `test.py`: inference script to generate a caption from an image using saved checkpoints
- `requirements.txt`: Python dependencies
- `checkpoints/vlm_distill/`: saved outputs (ignored by git)

## How It Works

1. The vision encoder extracts image tokens from SigLIP.
2. A linear projector maps vision hidden states to the language model hidden size.
3. Projected vision tokens are concatenated with text token embeddings.
4. The LM is trained with caption targets, while vision token labels are masked with `-100`.

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -U pip
pip install -r requirements.txt
```

Notes:

- `bitsandbytes` is used for 4-bit QLoRA loading on GPU.
- If you are CPU-only, you may need to remove quantization usage in code for stable runtime.

## Training

Run:

```bash
python vlm_distill.py
```

Current defaults in `main()`:

- Dataset: `lmms-lab/LLaVA-ReCap-118K`
- Batch size: `4`
- Epochs: `2`
- Learning rate: `1e-4`

## Saved Checkpoints

After training, artifacts are written to:

- `checkpoints/vlm_distill/language_adapter/`
- `checkpoints/vlm_distill/projector.pt`

Contents:

- LoRA adapter weights and config (PEFT format)
- Tokenizer files used by the fine-tuned LM
- Projector state dict + metadata (`vision_encoder_name`, `language_model_name`)

## Caption Inference

Use the saved model to caption an image:

```bash
python test.py --image images/bowl.jpg
```

Optional flags:

```bash
python test.py \
	--image images/bowl.jpg \
	--checkpoint-dir checkpoints/vlm_distill \
	--prompt "Describe this image in one concise caption." \
	--max-new-tokens 64
```

## Image Size Requirements

No manual resizing is required. The SigLIP processor resizes/preprocesses input images automatically.

## Troubleshooting

- Out of memory:
	- Reduce `batch_size` in `main()`.
	- Use fewer epochs for quick checks.
- Slow data loading:
	- First run may spend time downloading dataset/model files.
- Poor captions:
	- Train longer and/or clean filtering for noisy samples.

## Notes

- `checkpoints/` is ignored in `.gitignore`.
- Saved artifacts are sufficient to run inference with `test.py` after training.
