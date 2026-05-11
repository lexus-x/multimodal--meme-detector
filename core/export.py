import argparse
import os
import torch
from core.models import MultimodalClassifier
from core.dataset import load_glove_embeddings

def parse_args():
    parser = argparse.ArgumentParser(description="Export trained PyTorch model to ONNX and TorchScript for edge deployment.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the .pth checkpoint file")
    parser.add_argument("--glove_path", type=str, default="glove.6B/glove.6B.50d.txt")
    parser.add_argument("--output_dir", type=str, default="research/exports")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading checkpoint from {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    
    # Extract config from checkpoint
    ckpt_args = checkpoint.get("args", {})
    fusion_type = ckpt_args.get("fusion", "cross_attention")
    text_encoder = ckpt_args.get("text_encoder", "bilstm")
    text_hidden_dim = ckpt_args.get("text_hidden", 128)
    img_hidden_dim = ckpt_args.get("img_hidden", 256)
    img_backbone = ckpt_args.get("img_backbone", "vgg16")
    glove_dim = checkpoint.get("glove_dim", 50)
    
    print(f"Model Configuration: Fusion={fusion_type}, TextEncoder={text_encoder}, Backbone={img_backbone}")
    
    # Load embeddings to initialize model structure properly
    try:
        vocab, embeddings = load_glove_embeddings(args.glove_path, glove_dim)
        print(f"Loaded GloVe embeddings from {args.glove_path}")
    except Exception as e:
        print(f"Warning: Could not load GloVe embeddings from {args.glove_path}. Using random embeddings for export trace. ({e})")
        # Fallback to random embeddings for trace structure
        vocab_size = checkpoint.get("vocab_size", 400000)
        embeddings = torch.randn((vocab_size, glove_dim))
    
    # Initialize the model
    model = MultimodalClassifier(
        embedding_matrix=embeddings,
        text_hidden_dim=text_hidden_dim,
        text_encoder=text_encoder,
        img_hidden_dim=img_hidden_dim,
        img_backbone=img_backbone,
        fusion_type=fusion_type,
    )
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("Model weights loaded successfully.")
    
    # Create dummy inputs for tracing
    # (batch_size=1, max_seq_len=50) for text
    dummy_text = torch.randint(0, embeddings.shape[0], (1, 50), dtype=torch.long)
    # (batch_size=1, channels=3, height=224, width=224) for image
    dummy_image = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Export to TorchScript
    ts_path = os.path.join(args.output_dir, "model.pt")
    print(f"Exporting to TorchScript: {ts_path} ...")
    try:
        # We use tracing since the model is static. If it has dynamic control flow, use torch.jit.script
        traced_model = torch.jit.trace(model, (dummy_text, dummy_image))
        torch.jit.save(traced_model, ts_path)
        print("✓ TorchScript export successful.")
    except Exception as e:
        print(f"✗ TorchScript export failed: {e}")

    # 2. Export to ONNX
    onnx_path = os.path.join(args.output_dir, "model.onnx")
    print(f"Exporting to ONNX: {onnx_path} ...")
    try:
        torch.onnx.export(
            model,
            (dummy_text, dummy_image),
            onnx_path,
            export_params=True,
            opset_version=14,          # High enough for modern ops
            do_constant_folding=True,
            input_names=['input_ids', 'pixel_values'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'pixel_values': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            }
        )
        print("✓ ONNX export successful.")
    except Exception as e:
        print(f"✗ ONNX export failed: {e}")
        
    print(f"\nDone! Edge-ready models are available in {args.output_dir}/")

if __name__ == "__main__":
    main()
