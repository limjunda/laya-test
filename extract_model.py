"""Extract Laya English, Multilingual, and Typed-Decisions models from multi-part split zip archives in models/laya_model_parts/.

Usage:
    python extract_model.py                         # Extracts all three model checkpoints
    python extract_model.py --model english         # Extracts English model only
    python extract_model.py --model multilingual    # Extracts Multilingual model only
    python extract_model.py --model typed-decisions # Extracts Typed-Decisions model only
"""
import argparse
import io
from pathlib import Path
import zipfile


def extract_split_parts(pattern: str, parts_dir: Path, out_path: Path):
    part_files = sorted(parts_dir.glob(pattern))
    if not part_files:
        raise FileNotFoundError(f"No multi-part files matching '{pattern}' in {parts_dir}")

    print(f"\nReconstructing archive for {pattern} ({len(part_files)} parts)...")
    combined_buffer = io.BytesIO()
    total_bytes = 0
    for pf in part_files:
        print(f"  Reading {pf.name} ({pf.stat().st_size / 1024 / 1024:.2f} MB)...")
        with open(pf, "rb") as f:
            data = f.read()
            combined_buffer.write(data)
            total_bytes += len(data)

    print(f"Total reconstructed size: {total_bytes / 1024 / 1024:.2f} MB. Extracting to {out_path}...")
    out_path.mkdir(parents=True, exist_ok=True)
    combined_buffer.seek(0)
    with zipfile.ZipFile(combined_buffer, "r") as zf:
        zf.extractall(out_path)
        for member in zf.namelist():
            print(f"  Extracted: {member}")
    print(f"Successfully extracted to: {out_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Extract Laya checkpoints from split zip parts")
    parser.add_argument(
        "--model",
        choices=["all", "english", "multilingual", "typed-decisions"],
        default="all",
        help="Which model checkpoint to extract (default: all)"
    )
    parser.add_argument("--parts-dir", default="models/laya_model_parts", help="Directory containing split zip parts")
    parser.add_argument("--models-dir", default="models", help="Destination base directory")
    args = parser.parse_args()

    parts_path = Path(args.parts_dir)
    base_models_path = Path(args.models_dir)

    if args.model in ("all", "english"):
        extract_split_parts("laya_model_english.zip.*", parts_path, base_models_path / "laya_english")

    if args.model in ("all", "multilingual"):
        extract_split_parts("laya_model_multilingual.zip.*", parts_path, base_models_path / "laya_multilingual")

    if args.model in ("all", "typed-decisions"):
        extract_split_parts("laya_model_typed_decisions.zip.*", parts_path, base_models_path / "laya_typed_decisions")

    print("\nExtraction complete! You can load them locally with:")
    print("    import laya")
    print("    from laya import Router")
    print("    # Option A: Direct Agent load")
    print("    agent_en = laya.load('models/laya_english')")
    print("    agent_ml = laya.load('models/laya_multilingual')")
    print("    agent_td = laya.load('models/laya_typed_decisions')")
    print("    # Option B: Router with local paths")
    print("    router = Router(models={")
    print("        'english': 'models/laya_english',")
    print("        'multilingual': 'models/laya_multilingual',")
    print("        'typed-decisions': 'models/laya_typed_decisions'")
    print("    }, preload=True)")


if __name__ == "__main__":
    main()
