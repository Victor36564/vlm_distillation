import torch
from torch import nn
import requests
from io import BytesIO
from pathlib import Path
from PIL import Image

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModel, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ----- CONFIGS -----
VISION_ENCODER_NAME = "google/siglip-base-patch16-256"
LANGUAGE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

DATASET = "lmms-lab/LLaVA-ReCap-118K"
OUTPUT_DIR = "checkpoints/vlm_distill"

class Dataset(torch.utils.data.Dataset):
    def __init__(self, DATASET, vision_processor, language_tokenizer, split, max_length=512):
        self.dataset = load_dataset(DATASET)
        self.vision_processor = vision_processor
        self.language_tokenizer = language_tokenizer

        train_test_split = self.dataset["train"].train_test_split(test_size=0.2)
        self.train_data = train_test_split['train']
        self.test_data = train_test_split['test']

        if split == "train":
            self.dataset = self.train_data
        else:
            self.dataset = self.test_data

        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Process image
        image = item["image"]
        if isinstance(image, str):
            # If the image is a URL/path-like string, download and decode as PIL.
            try:
                response = requests.get(image, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
            except Exception:
                image = Image.new("RGB", (256, 256), color=0)
        elif isinstance(image, dict):
            # Some datasets store images as {"bytes": ..., "path": ...}
            if image.get("bytes") is not None:
                image = Image.open(BytesIO(image["bytes"])).convert("RGB")
            elif image.get("path") is not None:
                image = Image.open(image["path"]).convert("RGB")
            else:
                image = Image.new("RGB", (256, 256), color=0)
        else:
            # PIL images from datasets may have non-RGB mode.
            if hasattr(image, "convert"):
                image = image.convert("RGB")

        if image is None or not hasattr(image, "size"):
            image = Image.new("RGB", (256, 256), color=0)

        pixel_values = self.vision_processor(images=image, return_tensors="pt").pixel_values
        if pixel_values is None:
            # Fallback to a black image if processor fails for a corrupted sample.
            fallback = Image.new("RGB", (256, 256), color=0)
            pixel_values = self.vision_processor(images=fallback, return_tensors="pt").pixel_values
        image = pixel_values.squeeze(0)

        # Extract GPT caption from conversations
        conversations = item["conversations"]
        gpt_reply = next((c["value"] for c in conversations if c["from"] == "gpt"), None)
        if gpt_reply is None:
            gpt_reply = ""  # fallback

        # Tokenize caption
        text_inputs = self.language_tokenizer(
            gpt_reply,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = text_inputs.input_ids.squeeze(0)
        attention_mask = text_inputs.attention_mask.squeeze(0)

        return image, input_ids, attention_mask

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

def load_vision_encoder(model_name):
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype="auto",
    )

    # SigLIP checkpoints load as a dual encoder; keep only the vision branch.
    if hasattr(model, "vision_model"):
        model = model.vision_model

    print(f"Vision Encoder is running on: {next(model.parameters()).device}")

    processor = AutoProcessor.from_pretrained(model_name)

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
        torch_dtype="auto",
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, peft_config)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

class VLMModel(torch.nn.Module):
    def __init__(self, vision_encoder, language_model):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_model = language_model

        vision_hidden_size = getattr(vision_encoder.config, "hidden_size", None)
        if vision_hidden_size is None and hasattr(vision_encoder.config, "vision_config"):
            vision_hidden_size = vision_encoder.config.vision_config.hidden_size

        if vision_hidden_size is None:
            raise ValueError("Could not infer vision hidden size from vision encoder config.")

        self.projector = nn.Linear(vision_hidden_size, language_model.config.hidden_size)

    def forward(self, images, input_ids, attention_mask, labels=None):
        vision_device = next(self.vision_encoder.parameters()).device
        vision_dtype = next(self.vision_encoder.parameters()).dtype
        with torch.no_grad():
            vision_outputs = self.vision_encoder(pixel_values=images.to(device=vision_device, dtype=vision_dtype))
            vision_features = vision_outputs.last_hidden_state

        projected_features = self.projector(vision_features.to(DEVICE))

        text_embeddings = self.language_model.get_input_embeddings()(input_ids)

        input_embeddings = torch.cat([projected_features, text_embeddings], dim=1)

        vision_mask = torch.ones(projected_features.shape[:2], device=attention_mask.device)
        attention_mask = torch.cat([vision_mask, attention_mask], dim=1)

        if labels is not None:
            # We don't want the model to calculate loss on predicting the image itself
            # -100 is the PyTorch ignore_index for CrossEntropyLoss
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

def train(model, train_dataset, tokenizer, batch_size=1, lr=1e-4, epochs=1, device="cuda"):
    from torch.utils.data import DataLoader
    from torch.optim import AdamW
    from tqdm import tqdm

    model.train()
    model.to(device)

    # Only trainable parameters: projector + LoRA
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for images, input_ids, attention_mask in loop:
            # Move inputs to device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            images = images  # keep on CPU

            # Forward pass
            outputs = model(images, input_ids, attention_mask, labels=input_ids)
            loss = outputs.loss

            # Backward + optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())


def save_vlm_model(model, language_tokenizer, vision_encoder_name, language_model_name, output_dir=OUTPUT_DIR):
    output_path = Path(output_dir)
    adapter_path = output_path / "language_adapter"
    output_path.mkdir(parents=True, exist_ok=True)
    adapter_path.mkdir(parents=True, exist_ok=True)

    # Save LoRA adapter and tokenizer in Hugging Face format.
    model.language_model.save_pretrained(str(adapter_path))
    language_tokenizer.save_pretrained(str(adapter_path))

    # Save custom multimodal bridge weights and metadata.
    projector_payload = {
        "projector_state_dict": model.projector.state_dict(),
        "vision_encoder_name": vision_encoder_name,
        "language_model_name": language_model_name,
    }
    torch.save(projector_payload, output_path / "projector.pt")

    print(f"Saved VLM artifacts to: {output_path.resolve()}")

def main():

    torch.cuda.empty_cache()
    encoder_name = VISION_ENCODER_NAME
    vision_encoder, vision_tokenizer = load_vision_encoder(encoder_name)

    model_name = LANGUAGE_MODEL_NAME
    language_model, language_tokenizer = load_language_model(model_name)

    train_dataset = Dataset(DATASET, vision_tokenizer, language_tokenizer, split="train")
    test_dataset = Dataset(DATASET, vision_tokenizer, language_tokenizer, split="test")

    model = VLMModel(vision_encoder, language_model)
    model.projector.to(DEVICE)

    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Total trainable parameters: {trainable_params:,}")

    train(model, train_dataset, language_tokenizer, batch_size=4, lr=1e-4, epochs=2, device=DEVICE)

    save_vlm_model(
        model=model,
        language_tokenizer=language_tokenizer,
        vision_encoder_name=encoder_name,
        language_model_name=model_name,
    )

    # prompt = "Give me a short introduction to large language model."
    # messages = [
    #     {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    #     {"role": "user", "content": prompt}
    # ]
    # text = language_tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True
    # )
    # model_inputs = language_tokenizer([text], return_tensors="pt").to(language_model.device)

    # generated_ids = language_model.generate(
    #     **model_inputs,
    #     max_new_tokens=512
    # )
    # generated_ids = [
    #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    # ]

    # response = language_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # print(response)

if __name__ == "__main__":
    main()
