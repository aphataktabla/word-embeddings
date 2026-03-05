#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


def iter_txt_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*.txt")):
        if p.is_file():
            yield p


def parse_pair_line_first_option(line: str) -> Tuple[str, str]:
    """
    Input: "word1 word2" where word2 may be "x/y/z"
    Output: (word1, first_option_of_word2)
    """
    parts = line.strip().split()
    if len(parts) < 2:
        raise ValueError(f"Bad pair line (need 2 tokens): {line!r}")
    w1 = parts[0]
    w2 = parts[1]
    first = w2.split("/")[0].strip()
    if not first:
        raise ValueError(f"Empty first RHS option in line: {line!r}")
    return w1, first


def load_pairs(path: Path) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        out.append(parse_pair_line_first_option(s))
    return out


@dataclass(frozen=True)
class Quadruple:
    a: str
    b: str
    c: str
    d: str

    @property
    def analogy(self) -> str:
        return f"{self.a}:{self.b}::{self.c}:{self.d}"


def generate_quads(pairs: List[Tuple[str, str]], max_quads: int = 500) -> List[Quadruple]:
    # Use pairs from lines 1&2 -> one quadruple, 3&4 -> one quadruple, ...
    n = len(pairs)
    n_even = n - (n % 2)
    quads: List[Quadruple] = []
    for i in range(0, n_even, 2):
        (a, b) = pairs[i]
        (c, d) = pairs[i + 1]
        quads.append(Quadruple(a=a, b=b, c=c, d=d))
        if len(quads) >= max_quads:
            break
    return quads


def main() -> None:
    ap = argparse.ArgumentParser(description="Create analogy quadruples CSV from BATS, using first '/' option only.")
    ap.add_argument("--bats_root", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--max_quads_per_file", type=int, default=500)
    ap.add_argument("--max_total_quads", type=int, default=2000)
    args = ap.parse_args()

    bats_root = Path(args.bats_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "idx", "a", "b", "c", "d", "analogy", "cosine", "status", "missing"])

        for txt in iter_txt_files(bats_root):
            pairs = load_pairs(txt)
            quads = generate_quads(pairs, max_quads=args.max_quads_per_file)
            for i, q in enumerate(quads):
                w.writerow([
                    str(txt.relative_to(bats_root)), i,
                    q.a, q.b, q.c, q.d,
                    q.analogy,
                    "", "", ""
                ])
                rows_written += 1
                if args.max_total_quads and rows_written >= args.max_total_quads:
                    print(f"Reached max_total_quads={args.max_total_quads}. Wrote {rows_written} rows to {out_csv}")
                    return

    print(f"Wrote {rows_written} rows to {out_csv}")


if __name__ == "__main__":
    main()
