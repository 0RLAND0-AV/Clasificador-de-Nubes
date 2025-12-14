"""Quick dataset sanity checks.

- Counts images per class under data/.
- Warns if any class has < --min-per-class images.

Usage:
  python check_dataset_counts.py --min-per-class 15
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".avif"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Count images per class in data/")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).parent / "data"),
        help="Root data directory (default: ./data)",
    )
    parser.add_argument("--min-per-class", type=int, default=15)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Data dir not found: {data_dir}")

    counts: Counter[str] = Counter()
    for class_dir in sorted([p for p in data_dir.iterdir() if p.is_dir()]):
        n = 0
        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                n += 1
        counts[class_dir.name] = n

    print("Class counts:")
    for cls, n in sorted(counts.items()):
        status = "OK" if n >= args.min_per_class else "LOW"
        print(f"  {cls:>3s}: {n:4d}  [{status}]")

    low = {cls: n for cls, n in counts.items() if n < args.min_per_class}
    if low:
        print("\nClasses below minimum:")
        for cls, n in sorted(low.items(), key=lambda kv: kv[0]):
            print(f"  {cls}: {n} (< {args.min_per_class})")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
