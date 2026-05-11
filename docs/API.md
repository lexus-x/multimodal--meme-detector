# API Reference

## Base URL
```
http://localhost:8000
```

## Endpoints

### `GET /health`
Health check endpoint.

**Response:**
```json
{"status": "ok", "models_loaded": ["cross_attention_bilstm"], "device": "cpu"}
```

### `GET /models`
List all loaded models.

**Response:**
```json
{
  "models": [
    {"name": "cross_attention_bilstm", "fusion": "cross_attention", "parameters": {"total": 13900000, "trainable": 5000000}}
  ]
}
```

### `POST /predict`
Classify a single meme.

**Parameters (multipart/form-data):**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | file | Yes | Meme image (JPEG/PNG) |
| `text` | string | Yes | Text content of the meme |
| `fusion` | string | No | Fusion strategy: `early`, `cross_attention`, `gated`, `bilinear` (default: `cross_attention`) |

**Response:**
```json
{
  "offensive_prob": 0.73,
  "non_offensive_prob": 0.27,
  "label": "Offensive",
  "confidence": 0.73
}
```

### `POST /predict/batch`
Classify multiple memes at once.

**Parameters (multipart/form-data):**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `images` | file[] | Yes | Meme images |
| `texts` | string[] | Yes | Text content (same order as images) |
| `fusion` | string | No | Fusion strategy |

**Response:**
```json
{
  "results": [
    {"text": "meme text", "offensive_prob": 0.8, "label": "Offensive", "confidence": 0.8},
    {"text": "another meme", "offensive_prob": 0.2, "label": "Non-offensive", "confidence": 0.8}
  ]
}
```

### `GET /model/{fusion_type}/info`
Get detailed info about a model.

**Path parameters:**
- `fusion_type`: `early`, `cross_attention`, `gated`, `bilinear`

**Response:**
```json
{
  "fusion_type": "cross_attention",
  "total_parameters": 13900000,
  "trainable_parameters": 5000000,
  "frozen_parameters": 8900000,
  "text_encoder": "bilstm",
  "text_output_dim": 256,
  "image_output_dim": 256,
  "fusion_output_dim": 256
}
```

## Error Responses

All endpoints return standard HTTP errors:
```json
{"detail": "Error message"}
```

## Usage Examples

### Python
```python
import requests

resp = requests.post(
    "http://localhost:8000/predict",
    files={"image": open("meme.jpg", "rb")},
    data={"text": "When you fix one bug and create ten more", "fusion": "cross_attention"},
)
print(resp.json())
```

### cURL
```bash
curl -X POST http://localhost:8000/predict \
  -F "image=@meme.jpg" \
  -F "text=When you fix one bug and create ten more" \
  -F "fusion=cross_attention"
```

### JavaScript
```javascript
const formData = new FormData();
formData.append('image', fileInput.files[0]);
formData.append('text', 'When you fix one bug and create ten more');
formData.append('fusion', 'cross_attention');

const resp = await fetch('http://localhost:8000/predict', { method: 'POST', body: formData });
const data = await resp.json();
```
