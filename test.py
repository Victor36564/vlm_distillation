import argparse
from pathlib import Path
import random

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig

# Import your updated classes and configs from the distillation script
from vlm_distill import VLMModel, Dataset, OUTPUT_DIR, DATASET, DEVICE

DEFAULT_CHECKPOINT_DIR = "checkpoints/vlm_distill"

def load_models(checkpoint_dir: str, device: str):
    checkpoint_path = Path(checkpoint_dir)
    adapter_dir = checkpoint_path
    projector_path = checkpoint_path / "projector.pt"

    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
    if not projector_path.exists():
        raise FileNotFoundError(f"Projector checkpoint not found: {projector_path}")

    payload = torch.load(projector_path, map_location="cpu")
    vision_model_name = payload["vision_encoder_name"]
    language_model_name = payload["language_model_name"]

    # Vision encoder (Updated for custom architectures)
    vision_encoder = AutoModel.from_pretrained(vision_model_name, torch_dtype="auto", trust_remote_code=True)
    if hasattr(vision_encoder, "vision_model"):
        vision_encoder = vision_encoder.vision_model
    vision_encoder = vision_encoder.to(device)
    vision_encoder.eval()
    for p in vision_encoder.parameters():
        p.requires_grad = False

    vision_processor = AutoProcessor.from_pretrained(vision_model_name, trust_remote_code=True)

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

def load_trained_model(output_dir, device="cuda"):
    """Loads the base models and injects the trained projector weights."""
    model_dir = Path(f"{output_dir}")
    projector_path = Path(f"{output_dir}/projector.pt")

    if not projector_path.exists():
        raise FileNotFoundError(f"Could not find trained projector at {projector_path}. Did training finish?")

    print("Loading saved projector metadata...")
    payload = torch.load(projector_path, map_location=device, weights_only=True)
    vision_encoder_name = payload["vision_encoder_name"]
    language_model_name = payload["language_model_name"]

    print(f"Loading base Vision Encoder: {vision_encoder_name}")
    vision_encoder = AutoModel.from_pretrained(vision_encoder_name, torch_dtype="auto", trust_remote_code=True)
    if hasattr(vision_encoder, "vision_model"):
        vision_encoder = vision_encoder.vision_model
    vision_processor = AutoProcessor.from_pretrained(vision_encoder_name, trust_remote_code=True)

    print(f"Loading base Language Model: {language_model_name}")
    language_model = AutoModelForCausalLM.from_pretrained(language_model_name, torch_dtype="auto", device_map="auto")
    
    print("Loading saved tokenizer...")
    language_tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    if language_tokenizer.pad_token is None:
        language_tokenizer.pad_token = language_tokenizer.eos_token

    # Freeze base models (optional for inference, but saves memory)
    for param in vision_encoder.parameters(): param.requires_grad = False
    for param in language_model.parameters(): param.requires_grad = False

    print("Rebuilding VLM Model and injecting trained weights...")
    model = VLMModel(vision_encoder, language_model)
    
    # Load the trained weights into the projector
    model.projector.load_state_dict(payload["projector_state_dict"])
    model.projector.to(device)

    # Ensure dtype consistency
    model.projector.to(language_model.dtype)

    model.eval()
    return model, vision_processor, language_tokenizer

def test(model, test_dataset, vision_processor, tokenizer, device="cuda"):
    model.eval()
    model.to(device)

    print("--- Running Inference on a Random Test Sample ---")
    
    rand_idx = random.randint(0, len(test_dataset) - 1)
    raw_image, gt_text = test_dataset[rand_idx]
    
    # 1. Extract JUST the raw question (strip out "User:" and the Assistant's answer)
    if "\nAssistant:" in gt_text:
        raw_query = gt_text.split("\nAssistant:")[0].replace("User: ", "").strip()
    else:
        raw_query = gt_text
        
    # 2. Use Qwen's native chat template (This stops the hallucinations!)
    messages = [{"role": "user", "content": raw_query}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # 3. Tokenize the properly formatted prompt
    text_inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)
    
    # 4. Process the vision inputs dynamically
    vision_inputs = vision_processor(images=raw_image, return_tensors="pt")
    
    generated_ids = model.generate(
        vision_inputs=vision_inputs, 
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=50, 
        tokenizer=tokenizer
    )
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    print(f"Ground Truth Data:\n{gt_text.strip()}\n")
    print(f"Model Generated Answer:\n{generated_text.strip()}\n")

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

    # Update to support dynamic vision_inputs dictionary
    vision_inputs = vision_processor(images=image, return_tensors="pt")
    vision_inputs = {
        k: v.to(device=vision_device, dtype=vision_dtype) if v.is_floating_point() else v.to(device=vision_device) 
        for k, v in vision_inputs.items()
    }

    vision_outputs = vision_encoder(**vision_inputs)
    vision_features = vision_outputs.last_hidden_state
    projected_features = projector(vision_features.to(device=DEVICE, dtype=projector.weight.dtype))

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
    torch.cuda.empty_cache()

    # 1. Load the fully trained model
    model, vision_processor, language_tokenizer = load_trained_model(OUTPUT_DIR, device=DEVICE)

    # 2. Load the test dataset
    print(f"Loading dataset: {DATASET} (Test Split)")
    test_dataset = Dataset(DATASET, split="test")

    # 3. Run evaluation (Batch size removed since we dynamically process 1 at a time)
    test(model, test_dataset, vision_processor, language_tokenizer, device=DEVICE)

if __name__ == "__main__":
	main()

	# parser = argparse.ArgumentParser(description="Run VLM caption inference from saved checkpoint artifacts.")
	# parser.add_argument("--image", default="images/bowl.jpg", help="Path to input image")
	# parser.add_argument("--checkpoint-dir", default=OUTPUT_DIR, help="Checkpoint directory")
	# parser.add_argument("--prompt", default="Describe this image in one concise caption.", help="Caption prompt")
	# parser.add_argument("--max-new-tokens", type=int, default=64, help="Maximum generated tokens")
	# args = parser.parse_args()

	# device = "cuda" if torch.cuda.is_available() else "cpu"
	# model, vision_processor, language_tokenizer, projector, vision_encoder, language_model = load_trained_model(OUTPUT_DIR, device=DEVICE)

	# caption = caption_image(
	# 	image_path=args.image,
	# 	vision_encoder=vision_encoder,
	# 	vision_processor=vision_processor,
	# 	language_model=language_model,
	# 	tokenizer=language_tokenizer,
	# 	projector=projector,
	# 	prompt=args.prompt,
	# 	max_new_tokens=args.max_new_tokens,
	# )
	# print(caption)