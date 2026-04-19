import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import uvicorn
import io
import base64
from pathlib import Path

# Import your loading logic and configs from your test/distill scripts
from vlm_distill_textvqa import VLMModel, Dataset, DATASET, DEVICE, load_vision_encoder
from test_textvqa import load_trained_model

OUTPUT_DIR = "checkpoints/siglip-so400m-patch14-384__Qwen2.5-0.5B-Instruct__textvqacheckpoint"

# example payload
payload = {
    "prompt": "Summarize the image in one sentence.",
    "ocr_context": "",
    "image_path": "/home/vtishkev/vlm_distillation/images_test/bowl.jpg"
}

# --- 1. DEFINE API DATA STRUCTURES ---
class ChatRequest(BaseModel):
    prompt: str
    ocr_context: str = "a photo" # Default fallback for the CLIP text tower
    image_path: str = None       # Option A: Local file path
    image_base64: str = None     # Option B: Base64 encoded image

class ChatResponse(BaseModel):
    generated_text: str

# --- 2. INITIALIZE FASTAPI & GLOBALS ---
app = FastAPI(title="Custom VLM API")

# Global variables to hold the model in VRAM
vlm_model = None
vision_processor = None
language_tokenizer = None
clip_tokenizer = None

# --- 3. LOAD MODEL ON STARTUP ---
@app.on_event("startup")
async def startup_event():
    global vlm_model, vision_processor, language_tokenizer, clip_tokenizer
    print("Loading model into VRAM... (This will only happen once!)")
    
    # Unpack your custom loading function
    # NOTE: Ensure your load_trained_model returns the clip_tokenizer too!
    vlm_model, vision_processor, clip_tokenizer, language_tokenizer = load_trained_model(OUTPUT_DIR, device=DEVICE)
    print("Model successfully loaded and ready for queries!")

# --- 4. DEFINE THE INFERENCE ENDPOINT ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if vlm_model is None:
        raise HTTPException(status_code=503, detail="Model is still loading.")

    # 1. Load the Image
    try:
        if request.image_base64:
            image_data = base64.b64decode(request.image_base64)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        elif request.image_path:
            image = Image.open(request.image_path).convert("RGB")
        else:
            raise HTTPException(status_code=400, detail="Must provide either image_path or image_base64")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {str(e)}")

    vlm_model.eval()
    
    # 1. Fetch raw data
    # raw_image, raw_caption, raw_qa_text = dataset[idx]

    # # 2. Extract just the question to feed the model
    # if "\nAssistant:" in raw_qa_text:
    #     raw_query = raw_qa_text.split("\nAssistant:")[0].replace("User: ", "").strip()
    #     expected_answer = raw_qa_text.split("\nAssistant:")[1].strip()
    # else:
    #     raw_query = raw_qa_text
    #     expected_answer = "N/A"
    
    raw_image = image
    raw_query = request.prompt
    raw_caption = request.ocr_context

    # 3. Format using the strict Qwen chat template
    messages = [{"role": "user", "content": raw_query}]
    prompt_text = language_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 4. Tokenize Text
    text_inputs = language_tokenizer(prompt_text, return_tensors="pt")
    input_ids = text_inputs.input_ids.to(DEVICE)
    attention_mask = text_inputs.attention_mask.to(DEVICE)

    # 5. Process Image
    vision_inputs = vision_processor(images=raw_image, return_tensors="pt")
    encoder_text_inputs = clip_tokenizer(text=raw_caption, return_tensors="pt", padding=True, truncation=True)

    # 6. Generate Prediction
    generated_ids = vlm_model.generate(
        vision_inputs=vision_inputs,
        text_inputs=encoder_text_inputs,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=100,
        tokenizer=language_tokenizer
    )

    generated_text = language_tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    
    return ChatResponse(generated_text=generated_text)

if __name__ == "__main__":
    # Run the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)