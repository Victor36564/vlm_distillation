import argparse
from pathlib import Path

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig


DEFAULT_CHECKPOINT_DIR = "checkpoints/vlm_distill"


def load_models(checkpoint_dir: str, device: str):
	checkpoint_path = Path(checkpoint_dir)
	adapter_dir = checkpoint_path / "language_adapter"
	projector_path = checkpoint_path / "projector.pt"

	if not adapter_dir.exists():
		raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
	if not projector_path.exists():
		raise FileNotFoundError(f"Projector checkpoint not found: {projector_path}")

	payload = torch.load(projector_path, map_location="cpu")
	vision_model_name = payload["vision_encoder_name"]
	language_model_name = payload["language_model_name"]

	# Vision encoder
	vision_encoder = AutoModel.from_pretrained(vision_model_name, torch_dtype="auto")
	if hasattr(vision_encoder, "vision_model"):
		vision_encoder = vision_encoder.vision_model
	vision_encoder = vision_encoder.to(device)
	vision_encoder.eval()
	for p in vision_encoder.parameters():
		p.requires_grad = False

	vision_processor = AutoProcessor.from_pretrained(vision_model_name)

	# Language model + LoRA adapter
	if device == "cuda":
		bnb_config = BitsAndBytesConfig(
			load_in_4bit=True,
			bnb_4bit_use_double_quant=True,
			bnb_4bit_quant_type="nf4",
			bnb_4bit_compute_dtype=torch.bfloat16,
		)
		base_lm = AutoModelForCausalLM.from_pretrained(
			language_model_name,
			quantization_config=bnb_config,
			torch_dtype="auto",
			device_map="auto",
		)
	else:
		base_lm = AutoModelForCausalLM.from_pretrained(language_model_name, torch_dtype="auto")
		base_lm = base_lm.to(device)

	language_model = PeftModel.from_pretrained(base_lm, str(adapter_dir))
	language_model.eval()

	tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))

	vision_hidden_size = getattr(vision_encoder.config, "hidden_size", None)
	if vision_hidden_size is None and hasattr(vision_encoder.config, "vision_config"):
		vision_hidden_size = vision_encoder.config.vision_config.hidden_size

	if vision_hidden_size is None:
		raise ValueError("Could not infer vision hidden size from vision encoder config.")

	projector = torch.nn.Linear(vision_hidden_size, language_model.config.hidden_size)
	projector.load_state_dict(payload["projector_state_dict"])
	projector = projector.to(device)
	projector.eval()

	return vision_encoder, vision_processor, language_model, tokenizer, projector


@torch.no_grad()
def caption_image(
	image_path: str,
	vision_encoder,
	vision_processor,
	language_model,
	tokenizer,
	projector,
	prompt: str,
	max_new_tokens: int,
):
	image = Image.open(image_path).convert("RGB")

	vision_device = next(vision_encoder.parameters()).device
	vision_dtype = next(vision_encoder.parameters()).dtype
	lm_device = next(language_model.parameters()).device

	pixel_values = vision_processor(images=image, return_tensors="pt").pixel_values
	pixel_values = pixel_values.to(device=vision_device, dtype=vision_dtype)

	vision_features = vision_encoder(pixel_values=pixel_values).last_hidden_state
	projected_features = projector(vision_features.to(next(projector.parameters()).device))

	messages = [{"role": "user", "content": prompt}]
	prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
	text_inputs = tokenizer(prompt_text, return_tensors="pt")
	input_ids = text_inputs.input_ids.to(lm_device)
	attention_mask = text_inputs.attention_mask.to(lm_device)

	text_embeddings = language_model.get_input_embeddings()(input_ids)
	projected_features = projected_features.to(device=text_embeddings.device, dtype=text_embeddings.dtype)

	input_embeddings = torch.cat([projected_features, text_embeddings], dim=1)

	vision_mask = torch.ones(projected_features.shape[:2], device=attention_mask.device, dtype=attention_mask.dtype)
	full_attention_mask = torch.cat([vision_mask, attention_mask], dim=1)

	generated_ids = language_model.generate(
		inputs_embeds=input_embeddings,
		attention_mask=full_attention_mask,
		max_new_tokens=max_new_tokens,
		do_sample=False,
		temperature=1.0,
		pad_token_id=tokenizer.eos_token_id,
		eos_token_id=tokenizer.eos_token_id,
	)

	return tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()


def main():
	parser = argparse.ArgumentParser(description="Run VLM caption inference from saved checkpoint artifacts.")
	parser.add_argument("--image", required=True, help="Path to input image")
	parser.add_argument("--checkpoint-dir", default=DEFAULT_CHECKPOINT_DIR, help="Checkpoint directory")
	parser.add_argument("--prompt", default="Describe this image in one concise caption.", help="Caption prompt")
	parser.add_argument("--max-new-tokens", type=int, default=64, help="Maximum generated tokens")
	args = parser.parse_args()

	device = "cuda" if torch.cuda.is_available() else "cpu"
	vision_encoder, vision_processor, language_model, tokenizer, projector = load_models(args.checkpoint_dir, device)

	caption = caption_image(
		image_path=args.image,
		vision_encoder=vision_encoder,
		vision_processor=vision_processor,
		language_model=language_model,
		tokenizer=tokenizer,
		projector=projector,
		prompt=args.prompt,
		max_new_tokens=args.max_new_tokens,
	)
	print(caption)


if __name__ == "__main__":
	main()