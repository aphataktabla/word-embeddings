#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class VocabEntry:
    token: str
    token_id: int
    count: int


def load_vocab_tsv(vocab_path: str | Path) -> Dict[str, VocabEntry]:
    vocab_path = Path(vocab_path)
    token2entry: Dict[str, VocabEntry] = {}
    with vocab_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"Bad vocab.tsv line {line_no}: {line!r}")
            w, sid, sc = parts
            token2entry[w] = VocabEntry(token=w, token_id=int(sid), count=int(sc))
    return token2entry


def dense_cosine(u: np.ndarray, v: np.ndarray) -> float:
    dot = float(np.dot(u, v))
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot / (nu * nv)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Score analogy quadruples CSV using a dense embedding matrix where "
            "row i is the embedding for vocab id i."
        )
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="OUTDIR containing vocab.tsv",
    )
    ap.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Path to .npy embedding matrix, e.g. ppmi_svd_u.npy or cooc_svd_u.npy",
    )
    ap.add_argument(
        "--in_csv",
        type=str,
        required=True,
        help="Input CSV created by make_bats_quadruples.py or make_random_quadruples.py",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="Output CSV with cosine filled where possible",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    vocab_path = out_dir / "vocab.tsv"
    token2entry = load_vocab_tsv(vocab_path)

    embeddings = np.load(args.embeddings)
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be a 2D matrix, got shape {embeddings.shape}")

    rows: List[dict] = []
    with Path(args.in_csv).open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        required_cols = {"a", "b", "c", "d"}
        missing_cols = required_cols - set(r.fieldnames or [])
        if missing_cols:
            raise ValueError(f"Input CSV missing columns: {sorted(missing_cols)}")

        for row in r:
            a, b, c, d = row["a"], row["b"], row["c"], row["d"]
            missing_words = [w for w in (a, b, c, d) if w not in token2entry]

            if not missing_words:
                ids = [token2entry[w].token_id for w in (a, b, c, d)]
                missing_ids = [str(i) for i in ids if i < 0 or i >= embeddings.shape[0]]
                if missing_ids:
                    row["_status"] = "EMBED_OOB"
                    row["_missing"] = ",".join(missing_ids)
                else:
                    row["_status"] = "OK"
                    row["_missing"] = ""
            else:
                row["_status"] = "OOV"
                row["_missing"] = ",".join(missing_words)

            rows.append(row)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    base_fieldnames = list(rows[0].keys()) if rows else []
    base_fieldnames = [k for k in base_fieldnames if not k.startswith("_")]
    for col in ["cosine", "status", "missing"]:
        if col not in base_fieldnames:
            base_fieldnames.append(col)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=base_fieldnames)
        w.writeheader()

        for row in rows:
            if row["_status"] != "OK":
                row["cosine"] = ""
                row["status"] = row["_status"]
                row["missing"] = row["_missing"]
                w.writerow({k: row.get(k, "") for k in base_fieldnames})
                continue

            a, b, c, d = row["a"], row["b"], row["c"], row["d"]
            ia = token2entry[a].token_id
            ib = token2entry[b].token_id
            ic = token2entry[c].token_id
            id_ = token2entry[d].token_id

            v1 = embeddings[ia] - embeddings[ib]
            v2 = embeddings[ic] - embeddings[id_]
            cos = dense_cosine(v1, v2)

            row["cosine"] = f"{cos:.6f}"
            row["status"] = "OK"
            row["missing"] = ""
            w.writerow({k: row.get(k, "") for k in base_fieldnames})

    print(f"Loaded embeddings shape: {embeddings.shape}")
    print(f"Wrote scored CSV to {out_csv}")


if __name__ == "__main__":
    main()
