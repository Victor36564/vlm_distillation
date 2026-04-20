import torch
from torch import nn
from torch.utils.data import DataLoader
import requests
import re
from io import BytesIO
from pathlib import Path
from PIL import Image

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModel, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from tqdm import tqdm

# ----- CONFIGS -----
VISION_ENCODER_NAME = "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M" # "google/siglip-so400m-patch14-384" "google/siglip-base-patch16-384" "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
LANGUAGE_MODEL_NAME = "HuggingFaceTB/SmolLM-135M" # "Qwen/Qwen2.5-0.5B-Instruct" "openai-community/gpt2-medium" "MiniLLM/MiniPLM-Qwen-200M" "HuggingFaceTB/SmolLM-135M"
MODEL_FINETUNE = True

# 1. UPDATED DATASET REPOSITORY
DATASET = "liuhaotian/LLaVA-Pretrain"
IMAGE_BASE_DIR = Path("llava_images_100k")
OUTPUT_DIR = f"checkpoints/{VISION_ENCODER_NAME.split('/')[-1]}__{LANGUAGE_MODEL_NAME.split('/')[-1]}__LLaVA"

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

class Dataset(torch.utils.data.Dataset):
    def __init__(self, DATASET, split):
        # Load your newly created, perfectly matched 100k subset
        full_dataset = load_dataset("json", data_files="blip_100k_subset.json", split="train")
        
        # Split the 100k subset into Train (80k) and Test (20k)
        train_test_split = full_dataset.train_test_split(test_size=0.2, seed=42)
        
        if split == "train":
            self.dataset = train_test_split['train']
        else:
            self.dataset = train_test_split['test']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # FIX 2: Load the image directly from your local extracted folder
        image_path_str = item.get("image")
        
        if image_path_str:
            # Construct the absolute path to the image on your hard drive
            local_image_path = IMAGE_BASE_DIR / image_path_str
            try:
                image = Image.open(local_image_path).convert("RGB")
            except Exception as e:
                print(f"Warning: Could not load {local_image_path} - {e}")
                image = Image.new("RGB", (256, 256), color=0)
        else:
            image = Image.new("RGB", (256, 256), color=0)

        # ----- COORDINATE NORMALIZATION (Safely ignores non-coordinate text) -----
        width, height = image.size
        
        bbox_pattern = r"\[\(([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)\]"
        
        def normalize_match(match):
            x1, y1, x2, y2 = map(float, match.groups())
            nx1 = max(0, min(1000, int((x1 / width) * 1000)))
            ny1 = max(0, min(1000, int((y1 / height) * 1000)))
            nx2 = max(0, min(1000, int((x2 / width) * 1000)))
            ny2 = max(0, min(1000, int((y2 / height) * 1000)))
            return f"[({nx1}, {ny1}, {nx2}, {ny2})]"

        conversations = item.get("conversations", [])
        human_query = next((c["value"] for c in conversations if c["from"] == "human"), "")
        human_query = human_query.replace("<image>\n", "").replace("<image>", "").strip()
        
        gpt_reply = next((c["value"] for c in conversations if c["from"] == "gpt"), "")
        
        human_query = re.sub(bbox_pattern, normalize_match, human_query)
        gpt_reply = re.sub(bbox_pattern, normalize_match, gpt_reply)
        
        combined_text = f"User: {human_query}\nAssistant: {gpt_reply}"

        # print(item)

        return image, combined_text

def load_vision_encoder(model_name, freeze=True):
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    # print(model.config)

    print(f"Vision Encoder is running on: {next(model.parameters()).device}")

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    return model, processor

def load_language_model(model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # print(model.config)

    model = prepare_model_for_kbit_training(model)

    if LANGUAGE_MODEL_NAME in ["Qwen/Qwen2.5-0.5B-Instruct"]:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif LANGUAGE_MODEL_NAME in ["openai-community/gpt2-medium"]:
        target_modules = ["c_attn", "c_proj", "c_fc"]
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

class VLMModel(torch.nn.Module):
    def __init__(self, vision_encoder, language_model):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_model = language_model

        # print(vision_encoder.config)
        vision_hidden_size = getattr(vision_encoder.config, "hidden_size", None)
        if vision_hidden_size is None and hasattr(vision_encoder.config, "vision_config"):
            vision_hidden_size = vision_encoder.config.vision_config.hidden_size
        elif vision_hidden_size is None and hasattr(vision_encoder.config, "hidden_sizes"):
            vision_hidden_size = vision_encoder.config.hidden_sizes[-1]
        
        if vision_hidden_size is None:
            raise ValueError("Could not infer vision hidden size from vision encoder config.")

        self.projector = nn.Sequential(
            nn.Linear(vision_hidden_size, language_model.config.hidden_size),
            nn.GELU(),
            nn.Linear(language_model.config.hidden_size, language_model.config.hidden_size)
        )
        self.projector.to(torch.bfloat16)

    def forward(self, vision_inputs, input_ids, attention_mask, labels=None):
        vision_device = next(self.vision_encoder.parameters()).device
        vision_dtype = next(self.vision_encoder.parameters()).dtype
        
        vision_inputs = {k: v.to(DEVICE) for k, v in vision_inputs.items()}

        with torch.no_grad():
            vision_features = self.vision_encoder.vision_model(**vision_inputs).last_hidden_state

        projected_features = self.projector(vision_features.to(dtype=torch.bfloat16))

        text_embeddings = self.language_model.get_input_embeddings()(input_ids)

        input_embeddings = torch.cat([projected_features, text_embeddings], dim=1)

        vision_mask = torch.ones(projected_features.shape[:2], device=attention_mask.device)
        attention_mask = torch.cat([vision_mask, attention_mask], dim=1)

        if labels is not None:
            vision_labels = torch.full(
                projected_features.shape[:2], 
                -100, 
                device=labels.device, 
                dtype=labels.dtype
            )
            full_labels = torch.cat([vision_labels, labels], dim=1)
        else:
            full_labels = None

        return self.language_model(inputs_embeds=input_embeddings, attention_mask=attention_mask, labels=full_labels)
    
    def generate(self, vision_inputs, input_ids=None, attention_mask=None, max_new_tokens=100, tokenizer=None):
        vision_device = next(self.vision_encoder.parameters()).device
        vision_dtype = next(self.vision_encoder.parameters()).dtype
        
        vision_inputs = {
            k: v.to(device=vision_device, dtype=vision_dtype) if v.is_floating_point() else v.to(device=vision_device) 
            for k, v in vision_inputs.items()
        }
        
        with torch.no_grad():
            vision_outputs = self.vision_encoder(**vision_inputs)
            vision_features = vision_outputs.last_hidden_state
            projected_features = self.projector(vision_features.to(device=DEVICE, dtype=torch.bfloat16))

        if input_ids is not None and attention_mask is not None:
            with torch.no_grad():
                text_embeddings = self.language_model.get_input_embeddings()(input_ids)
            
            input_embeddings = torch.cat([projected_features, text_embeddings], dim=1)
            
            vision_mask = torch.ones(projected_features.shape[:2], device=attention_mask.device)
            full_attention_mask = torch.cat([vision_mask, attention_mask], dim=1)
        else:
            input_embeddings = projected_features
            full_attention_mask = torch.ones(projected_features.shape[:2], device=DEVICE)
        
        generated_ids = self.language_model.generate(
            inputs_embeds=input_embeddings,
            attention_mask=full_attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id if tokenizer else None,
            eos_token_id=tokenizer.eos_token_id if tokenizer else None,
            do_sample=False 
        )
        
        return generated_ids

def train(model, train_loader, vision_processor, language_tokenizer, lr=2e-4, epochs=1, accumulation_steps=8, device="cuda"):
    model.train()
    model.to(device)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    optimizer.zero_grad()
    epoch_bar = tqdm(range(epochs), desc="Epochs", unit="epoch")

    for epoch in epoch_bar:
        total_loss = 0.0
        n = 0
        best_loss = float("inf")
        for step_idx, (raw_image, raw_text) in enumerate(train_loader):
            vision_inputs = vision_processor(
                images=raw_image, 
                return_tensors="pt"
            )
            
            text_inputs = language_tokenizer(
                raw_text, 
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            input_ids = text_inputs.input_ids.to(device)
            attention_mask = text_inputs.attention_mask.to(device)

            outputs = model(vision_inputs, input_ids, attention_mask, labels=input_ids)
            
            loss = outputs.loss / accumulation_steps
            loss.backward()

            if (step_idx + 1) % accumulation_steps == 0 or (step_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += outputs.loss.item()
            n += 1

        epoch_bar.set_postfix(loss=total_loss / n)
        
        if (epoch + 1) % 1 == 0:
            if total_loss / n < best_loss:
                best_loss = total_loss / n

                save_vlm_model(
                    model=model,
                    language_tokenizer=language_tokenizer,
                    vision_encoder_name=VISION_ENCODER_NAME,
                    language_model_name=LANGUAGE_MODEL_NAME,
                    output_dir=OUTPUT_DIR + f"checkpoint"
                )

def save_vlm_model(model, language_tokenizer, vision_encoder_name, language_model_name, output_dir=OUTPUT_DIR):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # model.vision_encoder.save_pretrained(str(output_path))
    # vision_processor.save_pretrained(str(output_path))

    model.language_model.save_pretrained(str(output_path))
    language_tokenizer.save_pretrained(str(output_path))

    projector_payload = {
        "projector_state_dict": model.projector.state_dict(),
        "vision_encoder_name": vision_encoder_name,
        "language_model_name": language_model_name,
    }
    torch.save(projector_payload, output_path / "projector.pt")
    print(f"Saved VLM artifacts (LoRA + Projector) to: {output_path.resolve()}")

def collate_fn(batch):
    images = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    return images, texts

def main():
    torch.cuda.empty_cache()
    encoder_name = VISION_ENCODER_NAME
    vision_encoder, vision_processor = load_vision_encoder(encoder_name, freeze=True)

    model_name = LANGUAGE_MODEL_NAME
    language_model, language_tokenizer = load_language_model(model_name)

    train_dataset = Dataset(DATASET, split="train")
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, collate_fn=collate_fn)

    model = VLMModel(vision_encoder, language_model)
    model.projector.to(DEVICE)

    train(model, train_loader, vision_processor, language_tokenizer, lr=2e-4, epochs=50, device=DEVICE)

    save_vlm_model(
        model=model,
        language_tokenizer=language_tokenizer,
        vision_encoder_name=encoder_name,
        language_model_name=model_name,
    )

if __name__ == "__main__":
    main()