#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import List, Tuple


def load_vocab_words(vocab_tsv: Path) -> List[str]:
    """
    Reads vocab.tsv: word \t id \t count
    Returns list of words in file order.
    """
    words: List[str] = []
    with vocab_tsv.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"Bad vocab line: {line!r}")
            words.append(parts[0])
    return words


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate random analogy quadruples by sampling vocab words w/o replacement.")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Cooc OUTDIR containing vocab.tsv (e.g. /mnt/data/out/cc100_en_60M_v300k_w5)")
    ap.add_argument("--out_csv", type=str, required=True,
                    help="Output CSV path (e.g. /mnt/data/random_quadruples.csv)")
    ap.add_argument("--num_quads", type=int, default=1000,
                    help="Number of quadruples to generate (default 1000)")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for reproducibility (default 0)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    vocab_tsv = out_dir / "vocab.tsv"
    if not vocab_tsv.exists():
        raise FileNotFoundError(f"Missing {vocab_tsv}")

    words = load_vocab_words(vocab_tsv)
    V = len(words)

    need = args.num_quads * 4
    if need > V:
        raise ValueError(f"Need {need} unique words but vocab has only {V}. Reduce --num_quads.")

    rng = random.Random(args.seed)
    sample = rng.sample(words, k=need)  # without replacement

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "idx", "a", "b", "c", "d", "analogy", "cosine", "status", "missing"])

        for qi in range(args.num_quads):
            a, b, c, d = sample[4*qi : 4*qi + 4]
            analogy = f"{a}:{b}::{c}:{d}"
            w.writerow(["random_vocab_uniform", qi, a, b, c, d, analogy, "", "", ""])

    print(f"Wrote {args.num_quads} random quadruples to {out_csv}")


if __name__ == "__main__":
    main()
