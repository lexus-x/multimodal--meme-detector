"""
Download MultiOFF from HuggingFace and prepare CSV + images for core.train pipeline.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from datasets import load_dataset
from PIL import Image

OUTPUT_DIR = "data/processed"

def main():
    os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)

    print("Downloading MultiOFF dataset from HuggingFace...")
    ds = load_dataset("Ibrahim-Alam/multi-modal_offensive_meme")
    print(f"Splits: {list(ds.keys())}")

    for split_name in ds:
        split = ds[split_name]
        rows = []
        for i, item in enumerate(split):
            img = item["image"]
            label = int(item["label"])
            text = str(item.get("text", ""))
            fname = f"{split_name}_{i:04d}.jpg"
            img_path = os.path.join(OUTPUT_DIR, "images", fname)
            if not os.path.exists(img_path):
                img.convert("RGB").save(img_path)
            rows.append({"image_name": fname, "sentence": text, "label": label})

        df = pd.DataFrame(rows)
        # Map split names
        csv_name = split_name
        if csv_name == "validation":
            csv_name = "val"
        csv_path = os.path.join(OUTPUT_DIR, f"{csv_name}.csv")
        df.to_csv(csv_path, index=False)
        n_off = (df["label"] == 1).sum()
        print(f"  {split_name}: {len(df)} samples ({n_off} offensive) -> {csv_path}")

    print("\nDone! Dataset ready in data/processed/")

if __name__ == "__main__":
    main()
