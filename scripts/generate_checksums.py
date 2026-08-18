from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="SOURCE_SHA256.csv")
    parser.add_argument("--include-prepared", action="store_true")
    args = parser.parse_args()

    output = ROOT / args.output
    excluded_roots = {".git", ".venv", "venv", "data_raw", "release_assets"}
    if not args.include_prepared:
        excluded_roots.add("prepared")

    rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if rel.parts and rel.parts[0] in excluded_roots:
            continue
        if rel.as_posix() == args.output:
            continue
        if "__pycache__" in rel.parts or path.suffix in {".pyc", ".log", ".aux", ".out"}:
            continue
        rows.append({
            "relative_path": rel.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })

    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["relative_path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} checksum records to {output}")


if __name__ == "__main__":
    main()
