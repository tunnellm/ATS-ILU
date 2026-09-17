#!/usr/bin/env python3
"""Fetch one exact SuiteSparse group/name; extract only its primary matrix."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("matrix", help="exact group/name in the paper manifests")
    p.add_argument("--dest", type=Path, default=ROOT / "matrices")
    p.add_argument("--archive-sha256", help="optional previously verified download checksum")
    a = p.parse_args()
    rows = []
    for kind in ("unsymmetric", "spd"):
        with (ROOT / f"data/suitesparse/{kind}_scaling_or_manifest.csv").open() as f:
            rows += [dict(r, kind=kind) for r in csv.DictReader(f) if int(r["rows"]) >= 100000]
    matches = [r for r in rows if r["matrix"] == a.matrix]
    if len(matches) != 1:
        p.error("matrix must uniquely match the filtered manifests")
    row = matches[0]
    name = a.matrix.split("/")[1]
    dest = a.dest.resolve() / a.matrix.replace("/", "__")
    dest.mkdir(parents=True, exist_ok=True)
    final = dest / f"{name}.mtx"
    if final.exists():
        p.error(f"refusing to overwrite {final}")
    archive = dest / "matrix.tar.gz.part"
    # Keep the recorded URL rather than silently substituting another matrix.
    with urllib.request.urlopen(row["mm_url"], timeout=120) as source, archive.open("wb") as sink:
        shutil.copyfileobj(source, sink)
    with archive.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if a.archive_sha256 and digest != a.archive_sha256.lower():
        raise SystemExit("archive checksum mismatch; partial file retained")
    temporary = dest / f"{name}.mtx.part"
    with tarfile.open(archive) as tf:
        member = tf.getmember(f"{name}/{name}.mtx")
        if not member.isfile():
            raise SystemExit("primary matrix is not a regular archive member")
        with tf.extractfile(member) as source, temporary.open("wb") as sink:
            shutil.copyfileobj(source, sink)
    with temporary.open() as f:
        header = f.readline().split()
        if header[:3] != ["%%MatrixMarket", "matrix", "coordinate"]:
            raise SystemExit("unexpected Matrix Market format")
        dimensions = next(line for line in f if line.strip() and not line.startswith("%"))
        n, m, stored = map(int, dimensions.split())
        if n != int(row["rows"]) or m != n:
            raise SystemExit("matrix dimensions differ from recorded metadata")
    with temporary.open("rb") as f:
        matrix_hash = hashlib.file_digest(f, "sha256").hexdigest()
    row.update(archive_sha256=digest, matrix_sha256=matrix_hash, stored_entries=stored,
               historical_checksum_available=False)
    temporary.rename(final)
    (dest / "download.json").write_text(json.dumps(row, indent=2) + "\n")
    archive.unlink()
    print(final)
    print("Recorded new checksums. Historical download checksums were not retained; byte identity is not certified.")


if __name__ == "__main__":
    main()
