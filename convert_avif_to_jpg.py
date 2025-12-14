"""Convert AVIF images inside CloudClassify13/data to JPG.

- Reads .avif via pillow-avif-plugin (must be installed).
- Writes .jpg next to the original file.
- Optionally deletes originals with --delete-original.

Usage (Windows):
  python convert_avif_to_jpg.py
  python convert_avif_to_jpg.py --delete-original

This script is intentionally simple and project-local.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image  # pillow-avif-plugin enables AVIF decoding


SUPPORTED_INPUT_EXTS = {".avif"}


def convert_one(path: Path, *, delete_original: bool) -> Path:
    img = Image.open(path).convert("RGB")
    out_path = path.with_suffix(".jpg")
    img.save(out_path, format="JPEG", quality=95, optimize=True)

    if delete_original:
        path.unlink(missing_ok=True)

    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert .avif images to .jpg under data/")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).parent / "data"),
        help="Root data directory (default: ./data)",
    )
    parser.add_argument(
        "--delete-original",
        action="store_true",
        help="Delete .avif after successful conversion",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Data dir not found: {data_dir}")

    avifs = [p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_INPUT_EXTS]
    if not avifs:
        print("No .avif files found.")
        return 0

    print(f"Found {len(avifs)} .avif files")

    converted = 0
    for p in sorted(avifs):
        try:
            out = convert_one(p, delete_original=args.delete_original)
            print(f"OK  {p.relative_to(data_dir)} -> {out.relative_to(data_dir)}")
            converted += 1
        except Exception as exc:
            print(f"FAIL {p} ({exc})")

    print(f"Converted {converted}/{len(avifs)}")
    return 0 if converted == len(avifs) else 2


if __name__ == "__main__":
    raise SystemExit(main())
