"""
Multimodal Meme Detector — Local App
=====================================
Zero internet required. Single command:
    python run_app.py

Upload any meme image → automatic OCR text extraction → offensive content prediction.
"""

import os
import sys
import io
import base64
import time
import json
import logging
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

# ── Setup paths ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.dataset import load_glove_embeddings
from core.models import MultimodalClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("MemeDetector")

# ── Config ────────────────────────────────────────────────────────
GLOVE_PATH = str(ROOT / "data" / "glove.6B" / "glove.6B.50d.txt")
CKPT_DIR = ROOT / "checkpoints"
PORT = 7860
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ── Load model ────────────────────────────────────────────────────
def find_best_checkpoint():
    """Find the best checkpoint across all runs."""
    best_f1, best_path, best_name = -1, None, None
    for run_dir in CKPT_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        ckpt = run_dir / "best_model.pth"
        if ckpt.exists():
            data = torch.load(ckpt, map_location="cpu", weights_only=False)
            f1 = data.get("val_f1", 0)
            if f1 > best_f1:
                best_f1 = f1
                best_path = ckpt
                best_name = run_dir.name
    return best_path, best_name, best_f1


log.info(f"Device: {DEVICE}")
log.info("Loading GloVe embeddings...")
VOCAB, EMBEDDINGS = load_glove_embeddings(GLOVE_PATH, 50)

log.info("Finding best checkpoint...")
ckpt_path, ckpt_name, ckpt_f1 = find_best_checkpoint()
if ckpt_path is None:
    log.error("No checkpoint found in research/checkpoints/. Train a model first.")
    sys.exit(1)

log.info(f"Loading model: {ckpt_name} (val F1={ckpt_f1:.4f})")
ckpt_data = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
ckpt_args = ckpt_data.get("args", {})

MODEL = MultimodalClassifier(
    embedding_matrix=EMBEDDINGS,
    text_hidden_dim=ckpt_args.get("text_hidden", 128),
    text_encoder=ckpt_args.get("text_encoder", "bilstm"),
    img_hidden_dim=ckpt_args.get("img_hidden", 256),
    img_backbone=ckpt_args.get("img_backbone", "vgg16"),
    fusion_type=ckpt_args.get("fusion", "early"),
).to(DEVICE)
MODEL.load_state_dict(ckpt_data["model_state_dict"])
MODEL.eval()

params = MODEL.count_parameters()
log.info(f"Model loaded: {params['total']:,} params ({params['trainable']:,} trainable)")

# ── OCR ───────────────────────────────────────────────────────────
log.info("Loading EasyOCR (GPU)...")
import easyocr
OCR_READER = easyocr.Reader(["en"], gpu=torch.cuda.is_available(), verbose=False)
log.info("EasyOCR ready.")


def extract_text(image: Image.Image) -> str:
    """Extract text from image using EasyOCR."""
    img_np = np.array(image.convert("RGB"))
    results = OCR_READER.readtext(img_np, detail=0)
    return " ".join(results).strip()


def tokenize(text: str, max_len: int = 50) -> torch.Tensor:
    tokens = str(text).lower().split()
    indices = [VOCAB.get(t, VOCAB.get("<unk>", 1)) for t in tokens]
    if len(indices) < max_len:
        indices += [0] * (max_len - len(indices))
    else:
        indices = indices[:max_len]
    return torch.tensor(indices, dtype=torch.long)


@torch.no_grad()
def predict(image: Image.Image, text: str) -> dict:
    img_tensor = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    txt_tensor = tokenize(text).unsqueeze(0).to(DEVICE)
    t0 = time.time()
    logits = MODEL(txt_tensor, img_tensor)
    prob = torch.sigmoid(logits).item()
    elapsed = (time.time() - t0) * 1000
    return {
        "offensive_probability": round(prob, 4),
        "label": "Offensive" if prob > 0.5 else "Non-offensive",
        "confidence": round(max(prob, 1 - prob) * 100, 1),
        "inference_ms": round(elapsed, 1),
        "extracted_text": text,
    }


# ── HTML UI ───────────────────────────────────────────────────────
with open(ROOT / "web" / "ui_template.html", "r") as f:
    HTML_PAGE = f.read().replace("{{MODEL}}", ckpt_name).replace("{{DEVICE}}", str(DEVICE)).replace("{{F1}}", f"{ckpt_f1:.3f}")



# ── HTTP Server ───────────────────────────────────────────────────
class MemeHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        log.info(f"{self.address_string()} - {format % args}")

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "device": str(DEVICE), "model": ckpt_name}).encode())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/predict":
            content_type = self.headers.get("Content-Type", "")
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)

            if "multipart/form-data" in content_type:
                # Extract boundary
                boundary = None
                for part in content_type.split(";"):
                    part = part.strip()
                    if part.startswith("boundary="):
                        boundary = part.split("=", 1)[1].strip('"')
                if boundary:
                    boundary_bytes = boundary.encode()
                    parts = body.split(b"--" + boundary_bytes)
                    img_data = None
                    for part in parts:
                        if b"Content-Disposition" in part and b'name="image"' in part:
                            # Split headers from body at double newline
                            if b"\r\n\r\n" in part:
                                img_data = part.split(b"\r\n\r\n", 1)[1].rstrip(b"\r\n--")
                            elif b"\n\n" in part:
                                img_data = part.split(b"\n\n", 1)[1].rstrip(b"\n--")
                            break
                    if img_data is None:
                        self.send_response(400)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"error": "No image field found"}).encode())
                        return
                else:
                    img_data = body
            else:
                img_data = body

            try:
                image = Image.open(io.BytesIO(img_data)).convert("RGB")
                log.info(f"Image received: {image.size}")

                # OCR
                t0 = time.time()
                text = extract_text(image)
                ocr_ms = (time.time() - t0) * 1000
                log.info(f"OCR ({ocr_ms:.0f}ms): '{text[:80]}...' " if len(text) > 80 else f"OCR ({ocr_ms:.0f}ms): '{text}'")

                # Predict
                result = predict(image, text if text else "no text detected")

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode())
            except Exception as e:
                log.error(f"Prediction error: {e}")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_error(404)


# ── Main ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), MemeHandler)
    print()
    print("╔════════════════════════════════════════════════════════╗")
    print("║         🛡️  MEME DETECTOR — READY                    ║")
    print("╠════════════════════════════════════════════════════════╣")
    print(f"║  URL:    http://localhost:{PORT}                      ║")
    print(f"║  Device: {str(DEVICE):44s} ║")
    print(f"║  Model:  {ckpt_name:44s} ║")
    print("║  OCR:    EasyOCR (auto text extraction)               ║")
    print("╠════════════════════════════════════════════════════════╣")
    print("║  Just upload any meme image — no internet needed!     ║")
    print("╚════════════════════════════════════════════════════════╝")
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()
