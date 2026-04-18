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
LANGUAGE_MODEL_NAME = "MiniLLM/MiniPLM-Qwen-200M" # "Qwen/Qwen2.5-0.5B-Instruct" "openai-community/gpt2-medium" "MiniLLM/MiniPLM-Qwen-200M" "HuggingFaceTB/SmolLM-135M"

# 1. UPDATED DATASET REPOSITORY
DATASET = "facebook/textvqa"
OUTPUT_DIR = f"checkpoints/{VISION_ENCODER_NAME.split('/')[-1]}__{LANGUAGE_MODEL_NAME.split('/')[-1]}__LLaVA"

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

class Dataset(torch.utils.data.Dataset):
    def __init__(self, DATASET, split):
        self.dataset = load_dataset(DATASET, split=split, trust_remote_code=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # 1. Extract Image
        image = item["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 2. Extract CLIP Text (The Context)
        ocr_tokens = item.get("ocr_tokens", [])
        
        # We need a fallback just in case the image has no text in it
        if len(ocr_tokens) > 0:
            raw_clip_text = " ".join(ocr_tokens)
        else:
            raw_clip_text = "a photo" 

        # 3. Extract QA Text (The LM Prompt)
        question = item.get("question", "")
        answers = item.get("answers", [])
        answer = answers[0] if len(answers) > 0 else "unanswerable"
        
        raw_qa_text = f"User: {question}\nAssistant: {answer}"

        # Return all THREE pieces of data
        return image, raw_clip_text, raw_qa_text

def load_vision_encoder(model_name, freeze=True):
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    # print(model.config)

    print(f"Vision Encoder is running on: {next(model.parameters()).device}")

    vision_processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    clip_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # freeze the vision encoder
    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    return model, vision_processor, clip_tokenizer

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

        vision_hidden_size = vision_encoder.config.vision_config.hidden_size
        text_hidden_size = vision_encoder.config.text_config.hidden_size
        language_hidden_size = language_model.config.hidden_size

        # vision projector
        self.vision_projector = nn.Sequential(
            nn.Linear(vision_hidden_size, language_hidden_size),
            nn.GELU(),
            nn.Linear(language_hidden_size, language_hidden_size)
        )
        self.vision_projector.to(torch.bfloat16)

        # text projector
        self.text_projector = nn.Sequential(
            nn.Linear(text_hidden_size, language_hidden_size),
            nn.GELU(),
            nn.Linear(language_hidden_size, language_hidden_size)
        )
        self.text_projector.to(torch.bfloat16)

    def forward(self, vision_inputs, text_inputs, input_ids, attention_mask, labels=None):
        vision_device = next(self.vision_encoder.parameters()).device
        vision_dtype = next(self.vision_encoder.parameters()).dtype
        
        vision_inputs = {k: v.to(DEVICE) for k, v in vision_inputs.items()}

        text_inputs = {k: v.to(DEVICE) for k, v in text_inputs.items()}

        with torch.no_grad():
            vision_features = self.vision_encoder.vision_model(**vision_inputs).last_hidden_state
        projected_vision = self.vision_projector(vision_features.to(dtype=torch.bfloat16))

        with torch.no_grad():
            text_features = self.vision_encoder.text_model(**text_inputs).last_hidden_state
        projected_text = self.text_projector(text_features.to(dtype=torch.bfloat16))

        text_embeddings = self.language_model.get_input_embeddings()(input_ids)

        input_embeddings = torch.cat([projected_vision, projected_text, text_embeddings], dim=1)

        vision_mask = torch.ones(projected_vision.shape[:2], device=attention_mask.device)
        text_mask = torch.ones(projected_text.shape[:2], device=attention_mask.device)
        full_attention_mask = torch.cat([vision_mask, text_mask, attention_mask], dim=1)

        if labels is not None:
            vision_labels = torch.full(
                projected_vision.shape[:2], 
                -100, 
                device=labels.device, 
                dtype=labels.dtype
            )

            text_labels = torch.full(
                projected_text.shape[:2],
                -100,
                device=labels.device,
                dtype=labels.dtype
            )

            full_labels = torch.cat([vision_labels, text_labels, labels], dim=1)
        else:
            full_labels = None

        return self.language_model(inputs_embeds=input_embeddings, attention_mask=full_attention_mask, labels=full_labels)
    
    def generate(self, vision_inputs, text_inputs, input_ids=None, attention_mask=None, max_new_tokens=100, tokenizer=None):
        vision_device = next(self.vision_encoder.parameters()).device
        vision_dtype = next(self.vision_encoder.parameters()).dtype
        
        vision_inputs = {k: v.to(DEVICE) for k, v in vision_inputs.items()}

        text_inputs = {k: v.to(DEVICE) for k, v in text_inputs.items()}

        with torch.no_grad():
            vision_features = self.vision_encoder.vision_model(**vision_inputs).last_hidden_state
        projected_vision = self.vision_projector(vision_features.to(dtype=torch.bfloat16))

        with torch.no_grad():
            text_features = self.vision_encoder.text_model(**text_inputs).last_hidden_state
        projected_text = self.text_projector(text_features.to(dtype=torch.bfloat16))

        with torch.no_grad():
            text_embeddings = self.language_model.get_input_embeddings()(input_ids)

        input_embeddings = torch.cat([projected_vision, projected_text, text_embeddings], dim=1)

        vision_mask = torch.ones(projected_vision.shape[:2], device=attention_mask.device)
        text_mask = torch.ones(projected_text.shape[:2], device=attention_mask.device)
        full_attention_mask = torch.cat([vision_mask, text_mask, attention_mask], dim=1)
        
        generated_ids = self.language_model.generate(
            inputs_embeds=input_embeddings,
            attention_mask=full_attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id if tokenizer else None,
            eos_token_id=tokenizer.eos_token_id if tokenizer else None,
            do_sample=False 
        )
        
        return generated_ids

def train(model, train_loader, vision_processor, text_processor, language_tokenizer, lr=2e-4, epochs=1, accumulation_steps=8, device="cuda"):
    model.train()
    model.to(device)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    optimizer.zero_grad()
    
    epoch_bar = tqdm(range(epochs), desc="Epochs", unit="epoch")
    for epoch in epoch_bar:
        total_loss = 0.0
        n = 0
        best_loss = float("inf")

        for step_idx, (raw_image, raw_caption, raw_qa_text) in enumerate(train_loader):
            vision_inputs = vision_processor(
                images=raw_image, 
                return_tensors="pt"
            )

            encoder_text_inputs = text_processor(
                text=raw_caption,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            
            lm_text_inputs = language_tokenizer(
                raw_qa_text, 
                return_tensors="pt",
                padding=True,
                truncation=True
            )

            input_ids = lm_text_inputs.input_ids.to(device)
            attention_mask = lm_text_inputs.attention_mask.to(device)

            outputs = model(vision_inputs, encoder_text_inputs, input_ids, attention_mask, labels=input_ids)
            
            loss = outputs.loss / accumulation_steps
            loss.backward()

            if (step_idx + 1) % accumulation_steps == 0 or (step_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += outputs.loss.item()
            n += 1

        epoch_bar.set_postfix(loss=total_loss / n)
        
        if (epoch + 1) % 10 == 0:
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
        "vision_projector_state_dict": model.vision_projector.state_dict(),
        "text_projector_state_dict": model.text_projector.state_dict(),
        "vision_encoder_name": vision_encoder_name,
        "language_model_name": language_model_name,
    }
    torch.save(projector_payload, output_path / "projector.pt")
    print(f"Saved VLM artifacts (LoRA + Projector) to: {output_path.resolve()}")

def collate_fn(batch):
    # Unpack the 3-item tuple returned by __getitem__
    images = [item[0] for item in batch]
    clip_texts = [item[1] for item in batch]
    qa_texts = [item[2] for item in batch]
    
    return images, clip_texts, qa_texts

def main():
    torch.cuda.empty_cache()
    encoder_name = VISION_ENCODER_NAME
    vision_encoder, vision_processor, text_processor = load_vision_encoder(encoder_name, freeze=True)

    model_name = LANGUAGE_MODEL_NAME
    language_model, language_tokenizer = load_language_model(model_name)

    train_dataset = Dataset(DATASET, split="train")
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, collate_fn=collate_fn)

    model = VLMModel(vision_encoder, language_model)
    model.vision_projector.to(DEVICE)
    model.text_projector.to(DEVICE)

    train(model, train_loader, vision_processor, text_processor, language_tokenizer, lr=2e-4, epochs=50, device=DEVICE)

    save_vlm_model(
        model=model,
        language_tokenizer=language_tokenizer,
        vision_encoder_name=encoder_name,
        language_model_name=model_name,
    )

if __name__ == "__main__":
    main()