"""
Full-scale training: run all 4 fusion strategies on the real MultiOFF dataset.
"""
import subprocess, sys, os

GLOVE = "data/glove.6B/glove.6B.50d.txt"
TRAIN = "data/processed/train.csv"
VAL   = "data/processed/val.csv"
IMG   = "data/processed/images"
OUT   = "research/checkpoints"
EPOCHS = 20
PATIENCE = 7

FUSIONS = [
    ("early",                "bilstm", "vgg16"),
    ("cross_attention",      "bilstm", "vgg16"),
    ("gated",                "bilstm", "vgg16"),
    ("bilinear",             "bilstm", "vgg16"),
]

for fusion, enc, bb in FUSIONS:
    name = f"{fusion}_{enc}_{bb}"
    print(f"\n{'='*70}")
    print(f"  TRAINING: {name}")
    print(f"{'='*70}")
    cmd = [
        sys.executable, "-m", "core.train",
        "--train_csv", TRAIN, "--val_csv", VAL, "--img_dir", IMG,
        "--glove_path", GLOVE, "--glove_dim", "50",
        "--text_encoder", enc, "--img_backbone", bb, "--fusion", fusion,
        "--epochs", str(EPOCHS), "--batch_size", "16", "--lr", "3e-4",
        "--patience", str(PATIENCE), "--augment",
        "--output_dir", OUT, "--run_name", name,
    ]
    subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)) + "/..")
