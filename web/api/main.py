from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
import io
import os
import logging
from core.dataset import load_glove_embeddings
from core.models import MultimodalClassifier
from torchvision import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Meme Detector API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load models (using dummy if no checkpoints)
VOCAB, EMBEDDINGS = load_glove_embeddings("research/data/glove.6B/glove.6B.50d.txt", 50)

MODELS = {}

def get_model(fusion_type="cross_attention"):
    if fusion_type not in MODELS:
        model = MultimodalClassifier(
            embedding_matrix=EMBEDDINGS,
            text_encoder="bilstm",
            fusion_type=fusion_type,
        ).to(DEVICE)
        
        ckpt_path = "research/checkpoints/best_run/best_model.pth"
        if os.path.exists(ckpt_path):
            try:
                ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
                # Filter out embedding weights if size mismatch (due to mock vocab vs full vocab)
                state_dict = ckpt["model_state_dict"]
                if state_dict["text_branch.embedding.weight"].shape != model.text_branch.embedding.weight.shape:
                    logger.info("Skipping embedding weights due to shape mismatch.")
                    state_dict.pop("text_branch.embedding.weight")
                model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded trained checkpoint: {ckpt_path}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        
        model.eval()
        MODELS[fusion_type] = model
    return MODELS[fusion_type]

def tokenize(text: str, vocab: dict, max_len: int = 50) -> torch.Tensor:
    tokens = str(text).lower().split()
    indices = [vocab.get(t, vocab.get("<unk>", 1)) for t in tokens]
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return torch.tensor(indices, dtype=torch.long)

@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": list(MODELS.keys()), "device": str(DEVICE)}

@app.get("/models")
async def list_models():
    """List all loaded models and their fusion types."""
    return {
        "models": [
            {"name": name, "fusion": getattr(m, "fusion_type", "unknown"), "parameters": m.count_parameters()}
            for name, m in MODELS.items()
        ]
    }

@app.get("/model/{fusion_type}/info")
async def model_info(fusion_type: str):
    """Get detailed info about a specific model."""
    model = get_model(fusion_type)
    params = model.count_parameters()
    return {
        "fusion_type": fusion_type,
        "total_parameters": params["total"],
        "trainable_parameters": params["trainable"],
        "frozen_parameters": params["frozen"],
        "text_encoder": model.text_branch.encoder_type,
        "text_output_dim": model.text_branch.output_dim,
        "image_output_dim": model.image_branch.output_dim,
        "fusion_output_dim": model.fusion.output_dim,
    }

@app.post("/predict/batch")
async def predict_batch(
    images: list[UploadFile] = File(...),
    texts: list[str] = Form(...),
    fusion: str = Form("cross_attention"),
):
    """Predict multiple memes at once."""
    if len(images) != len(texts):
        raise HTTPException(400, "Number of images must match number of texts")

    model = get_model(fusion)
    results = []

    for img_file, text in zip(images, texts):
        try:
            img_data = await img_file.read()
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            img_tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
            txt_tensor = tokenize(text, VOCAB).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(txt_tensor, img_tensor)
                prob = torch.sigmoid(logits).item()

            results.append({
                "text": text,
                "offensive_prob": prob,
                "label": "Offensive" if prob > 0.5 else "Non-offensive",
                "confidence": max(prob, 1 - prob),
            })
        except Exception as e:
            results.append({"text": text, "error": str(e)})

    return {"results": results}

@app.on_event("startup")
async def startup_event():
    """Preload default model on startup."""
    logger.info(f"Starting Meme Detector API on {DEVICE}")
    try:
        get_model("cross_attention")
        logger.info("Default model loaded")
    except Exception as e:
        logger.warning(f"Could not load default model: {e}")

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    with open("app/api/paper_ui.html", "r") as f:
        return f.read()

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    text: str = Form(...),
    fusion: str = Form("cross_attention")
):
    try:
        image_data = await image.read()
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        img_tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
        txt_tensor = tokenize(text, VOCAB).unsqueeze(0).to(DEVICE)
        
        # 1. Actual Model Prediction
        model = get_model(fusion)
        with torch.no_grad():
            logits = model(txt_tensor, img_tensor)
            model_prob = torch.sigmoid(logits).item()
        
        return {
            "offensive_prob": model_prob,
            "non_offensive_prob": 1 - model_prob,
            "label": "Offensive" if model_prob > 0.5 else "Non-offensive",
            "confidence": max(model_prob, 1 - model_prob)
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
