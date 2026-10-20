"""Extract Laya English model from multi-part split zip archives in models/laya_model_parts/.

Usage:
    python extract_model.py [--output-dir models/laya_english]
"""
import argparse
import io
import os
import zipfile
from pathlib import Path


def extract_laya_model(parts_dir: str = "models/laya_model_parts", output_dir: str = "models/laya_english"):
    parts_path = Path(parts_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    part_files = sorted(parts_path.glob("laya_model_english.zip.*"))
    if not part_files:
        raise FileNotFoundError(f"No multi-part files found in {parts_path} matching 'laya_model_english.zip.*'")

    print(f"Found {len(part_files)} zip parts in {parts_path}. Reconstructing archive...")
    combined_buffer = io.BytesIO()
    total_bytes = 0
    for pf in part_files:
        print(f"  Reading {pf.name} ({pf.stat().st_size / 1024 / 1024:.2f} MB)...")
        with open(pf, "rb") as f:
            data = f.read()
            combined_buffer.write(data)
            total_bytes += len(data)

    print(f"Reconstructed {total_bytes / 1024 / 1024:.2f} MB archive in memory. Extracting to {out_path}...")
    combined_buffer.seek(0)
    with zipfile.ZipFile(combined_buffer, "r") as zf:
        zf.extractall(out_path)
        for member in zf.namelist():
            print(f"  Extracted: {member}")

    print(f"\nSuccessfully extracted Laya model to: {out_path.resolve()}")
    print("You can now load it locally with:")
    print(f"    import laya")
    print(f"    agent = laya.load(r'{out_path.resolve()}')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Laya model from split zip parts")
    parser.add_argument("--parts-dir", default="models/laya_model_parts", help="Directory containing split zip parts")
    parser.add_argument("--output-dir", default="models/laya_english", help="Directory to extract the model into")
    args = parser.parse_args()
    extract_laya_model(args.parts_dir, args.output_dir)
