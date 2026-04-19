import torch
import random
from pathlib import Path
from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Import the architecture and configs from your main training script
# Ensure your training script is named 'vlm_distill.py'
from vlm_distill_LLaVA import VLMModel, Dataset, DATASET, DEVICE

MODEL_DIR = "checkpoints/siglip-so400m-patch14-384__MiniPLM-Qwen-200M__LLaVA"

def load_trained_model(output_dir=MODEL_DIR, device="cuda"):
    """
    Loads the base models, attaches the trained LoRA adapter to the language model,
    and injects the trained MLP projector weights.
    """
    model_dir = Path(output_dir)
    projector_path = model_dir / "projector.pt"

    if not projector_path.exists():
        raise FileNotFoundError(f"Could not find trained projector at {projector_path}. Did training finish?")

    print("Loading saved projector metadata...")
    payload = torch.load(projector_path, map_location="cpu", weights_only=True)
    vision_encoder_name = payload["vision_encoder_name"]
    language_model_name = payload["language_model_name"]

    # 1. Load Base Vision Encoder
    print(f"Loading base Vision Encoder: {vision_encoder_name}")
    vision_encoder = AutoModel.from_pretrained(vision_encoder_name, torch_dtype="auto", trust_remote_code=True)
    if hasattr(vision_encoder, "vision_model"):
        vision_encoder = vision_encoder.vision_model
    vision_processor = AutoProcessor.from_pretrained(vision_encoder_name, trust_remote_code=True)

    
    for param in vision_encoder.parameters(): 
        param.requires_grad = False

    # 2. Load Base Language Model (in 4-bit) and attach LoRA
    print(f"Loading base Language Model and attaching LoRA: {language_model_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    base_lm = AutoModelForCausalLM.from_pretrained(
        language_model_name, 
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    
    # Apply the fine-tuned LoRA weights we saved during training
    language_model = PeftModel.from_pretrained(base_lm, str(model_dir))
    language_model.eval()

    print("Loading saved tokenizer...")
    language_tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    if language_tokenizer.pad_token is None:
        language_tokenizer.pad_token = language_tokenizer.eos_token

    # 3. Rebuild VLM Architecture and inject MLP Projector weights
    print("Rebuilding VLM Model and injecting trained projector weights...")
    model = VLMModel(vision_encoder, language_model)
    
    model.projector.load_state_dict(payload["projector_state_dict"])
    model.projector.to(torch.bfloat16)
    model.projector.to(device)
    model.eval()

    return model, vision_processor, language_tokenizer

@torch.no_grad()
def test_single_sample(model, dataset, vision_processor, tokenizer, device="cuda", idx=None):
    """
    Tests a single sample from the dataset and prints the Ground Truth vs Model Output.
    """
    model.eval()

    if idx is None:
        idx = random.randint(0, len(dataset) - 1)

    print(f"\n" + "="*70)
    print(f"--- RUNNING INFERENCE ON TEST SAMPLE {idx} ---")
    
    # 1. Fetch raw data
    raw_image, gt_text = dataset[idx]

    # 2. Extract just the question to feed the model
    if "\nAssistant:" in gt_text:
        raw_query = gt_text.split("\nAssistant:")[0].replace("User: ", "").strip()
        expected_answer = gt_text.split("\nAssistant:")[1].strip()
    else:
        raw_query = gt_text
        expected_answer = "N/A"

    raw_query = "Is there a bowl in this image?"
    # 3. Format using the strict Qwen chat template
    messages = [{"role": "user", "content": raw_query}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 4. Tokenize Text
    text_inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)

    # 5. Process Image
    vision_inputs = vision_processor(images=raw_image, return_tensors="pt")

    # 6. Generate Prediction
    generated_ids = model.generate(
        vision_inputs=vision_inputs,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=40,
        tokenizer=tokenizer
    )

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    print(f"[User Question]:\n{raw_query}\n")
    print(f"[Ground Truth Answer (Normalized)]:\n{expected_answer}\n")
    print(f"[Model Generated Answer]:\n{generated_text}")
    print("="*70 + "\n")


def main():
    torch.cuda.empty_cache()

    # 1. Load the fully trained model (Base Models + LoRA + MLP Projector)
    model, vision_processor, language_tokenizer = load_trained_model(MODEL_DIR, device=DEVICE)

    # 2. Load the test split of the dataset
    print(f"\nLoading dataset: {DATASET} (Test Split)")
    test_dataset = Dataset(DATASET, split="train")

    # 3. Run evaluation on a random sample (or specify an idx to test a specific one)
    test_single_sample(model, test_dataset, vision_processor, language_tokenizer, device=DEVICE, idx=60482)

if __name__ == "__main__":
    main()