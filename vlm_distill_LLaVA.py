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
VISION_ENCODER_NAME = "google/siglip2-so400m-patch14-384" # "google/siglip2-so400m-patch14-384" "google/siglip-so400m-patch14-384" "google/siglip-base-patch16-384" "wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M" "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
LANGUAGE_MODEL_NAME = "MiniLLM/MiniPLM-Qwen-200M" # "Qwen/Qwen2.5-0.5B-Instruct" "openai-community/gpt2-medium" "MiniLLM/MiniPLM-Qwen-200M" "HuggingFaceTB/SmolLM-135M"

TRAINING_STAGE = 2  # Set to 1 for Pretraining, 2 for Instruct Tuning

if TRAINING_STAGE == 1:
    DATASET = "liuhaotian/LLaVA-Pretrain"
    DATASET_FILE = "blip_100k_subset.json"
    IMAGE_DIR = Path("llava_images_100k")
    OUTPUT_DIR = f"checkpoints/{VISION_ENCODER_NAME.split('/')[-1]}__{LANGUAGE_MODEL_NAME.split('/')[-1]}__Stage1"
else:
    DATASET = "liuhaotian/LLaVA-Instruct-150K"
    DATASET_FILE = "llava_instruct_150k.json"
    IMAGE_DIR = Path("./coco_images/train2017")
    OUTPUT_DIR = f"checkpoints/{VISION_ENCODER_NAME.split('/')[-1]}__{LANGUAGE_MODEL_NAME.split('/')[-1]}__Stage2"

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

class PreTrainDataset(torch.utils.data.Dataset):
    def __init__(self, DATASET, split):
        # Load your newly created, perfectly matched 100k subset
        full_dataset = load_dataset("json", data_files=DATASET_FILE, split="train")
        
        # Split the 100k subset into train/test (90k/10%)
        train_test_split = full_dataset.train_test_split(test_size=0.1, seed=42)
        
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
            local_image_path = IMAGE_DIR / image_path_str
            try:
                image = Image.open(local_image_path).convert("RGB")
            except Exception as e:
                print(f"Warning: Could not load {local_image_path} - {e}")
                image = Image.new("RGB", (256, 256), color=0)
        else:
            image = Image.new("RGB", (256, 256), color=0)

        conversations = item.get("conversations", [])
        human_query = next((c["value"] for c in conversations if c["from"] == "human"), "")
        human_query = human_query.replace("<image>\n", "").replace("<image>", "").strip()
        
        gpt_reply = next((c["value"] for c in conversations if c["from"] == "gpt"), "")
        
        combined_text = f"User: {human_query}\nAssistant: {gpt_reply}"

        # print(item)

        return image, combined_text

class InstructDataset(torch.utils.data.Dataset):
    def __init__(self, DATASET, split):
        # The 150k dataset repo has a specific JSON file we need
        full_dataset = load_dataset(DATASET, data_files=DATASET_FILE, split="train")
        
        train_test_split = full_dataset.train_test_split(test_size=0.1, seed=42)
        
        if split == "train":
            self.dataset = train_test_split['train']
        else:
            self.dataset = train_test_split['test']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # 1. Load the COCO Image
        image_file = item.get("image")
        if image_file:
            # LLaVA JSON often prefixes COCO images with 'train2017/' or similar, 
            # so we ensure we just grab the raw filename
            clean_filename = image_file.split("/")[-1]
            local_image_path = IMAGE_DIR / clean_filename
            
            try:
                image = Image.open(local_image_path).convert("RGB")
            except Exception as e:
                print(f"Warning: Could not load {local_image_path} - {e}")
                image = Image.new("RGB", (256, 256), color=0)
        else:
            # Some conversations in 150k are text-only!
            image = Image.new("RGB", (256, 256), color=0)

        # Parse MULTI-TURN Conversations
        conversations = item.get("conversations", [])
        formatted_qa_text = ""

        for turn in conversations:
            speaker = turn.get("from")
            text_value = turn.get("value", "")

            # Clean out the <image> tags that LLaVA inserts
            clean_text = text_value.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "").strip()

            if speaker == "human":
                formatted_qa_text += f"User: {clean_text}\n"
            elif speaker == "gpt":
                formatted_qa_text += f"Assistant: {clean_text}\n"

        # Trim trailing newlines
        raw_qa_text = formatted_qa_text.strip()

        # Return all 3 items for your Dual-Projector setup
        return image, raw_qa_text

def load_vision_encoder(model_name, freeze=True):
    model = AutoModel.from_pretrained(
        model_name,
        dtype="auto",
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
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    if TRAINING_STAGE == 1:
        # STAGE 1: Freeze the entire language model. No LoRA.
        for param in model.parameters():
            param.requires_grad = False
        print("Stage 1 Active: Language Model is FROZEN.")
        
    else:
        # STAGE 2: Apply LoRA to the language model.
        print("Stage 2 Active: Injecting LoRA adapters...")
        if LANGUAGE_MODEL_NAME in ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]:
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
            nn.Linear(vision_hidden_size, vision_hidden_size),
            nn.GELU(),
            nn.Linear(vision_hidden_size, language_model.config.hidden_size)
        )
        self.projector.to(torch.bfloat16)

    def forward(self, vision_inputs, input_ids, attention_mask, labels=None):
        vision_device = next(self.vision_encoder.parameters()).device
        vision_dtype = next(self.vision_encoder.parameters()).dtype
        
        vision_inputs = {k: v.to(DEVICE) for k, v in vision_inputs.items()}

        with torch.no_grad():
            vision_features = self.vision_encoder.get_image_features(**vision_inputs).last_hidden_state

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
            temperature=0.0,
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
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")
        for step_idx, (raw_image, raw_text) in enumerate(loop):
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
            loop.set_postfix(loss=total_loss / n)

        epoch_bar.set_postfix(loss=total_loss / n)
        
        if (epoch + 1) % 1 == 0 and (epoch + 1) != epochs:
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

    # Use the dynamic dataset configurations
    if TRAINING_STAGE == 1:
        train_dataset = PreTrainDataset(DATASET, split="train")
    else:
        train_dataset = InstructDataset(DATASET, split="train")
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    model = VLMModel(vision_encoder, language_model)
    
    if TRAINING_STAGE == 2:
        stage1_output_dir = f"checkpoints/{VISION_ENCODER_NAME.split('/')[-1]}__{LANGUAGE_MODEL_NAME.split('/')[-1]}__Stage1"
        projector_path = Path(stage1_output_dir) / "projector.pt"
        
        if projector_path.exists():
            print(f"Loading pre-trained Stage 1 Projector from {projector_path}...")
            payload = torch.load(projector_path, map_location="cpu", weights_only=True)
            model.projector.load_state_dict(payload["projector_state_dict"])
        else:
            print("WARNING: Could not find Stage 1 projector weights! Starting from scratch.")

    model.projector.to(DEVICE)

    # Stage 1 usually trains for 1 epoch. Stage 2 usually trains for 1 to 3 epochs.
    train_epochs = 1 if TRAINING_STAGE == 1 else 3
    train(model, train_loader, vision_processor, language_tokenizer, lr=2e-4, epochs=train_epochs, device=DEVICE)

    save_vlm_model(
        model=model,
        language_tokenizer=language_tokenizer,
        vision_encoder_name=encoder_name,
        language_model_name=model_name,
        output_dir=OUTPUT_DIR
    )

if __name__ == "__main__":
    main()