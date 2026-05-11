"""Data validation for the multimodal meme detector."""
import os
import pandas as pd
from PIL import Image
from pathlib import Path


def validate_dataset(csv_path: str, img_dir: str) -> dict:
    """Validate a dataset CSV and image directory.
    
    Returns dict with validation results and issues found.
    """
    issues = []
    stats = {"total": 0, "valid": 0, "missing_img": 0, "bad_img": 0, "bad_label": 0}
    
    if not os.path.exists(csv_path):
        return {"valid": False, "error": f"CSV not found: {csv_path}", "stats": stats, "issues": []}
    
    df = pd.read_csv(csv_path)
    
    # Check columns
    required = {"image_name", "sentence", "label"}
    missing = required - set(df.columns)
    if missing:
        return {"valid": False, "error": f"Missing columns: {missing}", "stats": stats, "issues": []}
    
    stats["total"] = len(df)
    
    for idx, row in df.iterrows():
        row_valid = True
        
        # Check label
        if row["label"] not in (0, 1):
            issues.append(f"Row {idx}: invalid label '{row['label']}' (expected 0 or 1)")
            stats["bad_label"] += 1
            row_valid = False
        
        # Check image exists
        img_path = os.path.join(img_dir, str(row["image_name"]))
        if not os.path.exists(img_path):
            issues.append(f"Row {idx}: image not found '{row['image_name']}'")
            stats["missing_img"] += 1
            row_valid = False
        else:
            # Check image is loadable
            try:
                img = Image.open(img_path)
                img.verify()
            except Exception as e:
                issues.append(f"Row {idx}: corrupt image '{row['image_name']}': {e}")
                stats["bad_img"] += 1
                row_valid = False
        
        # Check text
        if pd.isna(row["sentence"]) or str(row["sentence"]).strip() == "":
            issues.append(f"Row {idx}: empty sentence")
            row_valid = False
        
        if row_valid:
            stats["valid"] += 1
    
    return {
        "valid": stats["valid"] == stats["total"],
        "stats": stats,
        "issues": issues[:50],  # cap at 50
    }


def print_validation_report(result: dict):
    """Pretty-print validation results."""
    s = result["stats"]
    print(f"\n{'─' * 50}")
    print(f"  DATASET VALIDATION REPORT")
    print(f"{'─' * 50}")
    print(f"  Total rows:   {s['total']}")
    print(f"  Valid:        {s['valid']}")
    print(f"  Missing img:  {s['missing_img']}")
    print(f"  Bad images:   {s['bad_img']}")
    print(f"  Bad labels:   {s['bad_label']}")
    
    if result.get("error"):
        print(f"\n  ❌ Error: {result['error']}")
    elif result["valid"]:
        print(f"\n  ✅ Dataset is valid!")
    else:
        print(f"\n  ⚠️  {s['total'] - s['valid']} issues found")
        if result.get("issues"):
            print(f"\n  First issues:")
            for issue in result["issues"][:10]:
                print(f"    • {issue}")
    print(f"{'─' * 50}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--img_dir", required=True)
    args = parser.parse_args()
    
    result = validate_dataset(args.csv, args.img_dir)
    print_validation_report(result)
