#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

try:
    from scipy import sparse
except ImportError:  # optional unless --save_npz is used
    sparse = None


# Input cooc record format from cooc_cc100.cpp:
# uint32 i, uint32 j, uint64 c
COOC_REC_DTYPE = np.dtype([("i", "<u4"), ("j", "<u4"), ("c", "<u8")])

# Output PMI bucket record format:
# uint32 i, uint32 j, float32 pmi
PMI_REC_DTYPE = np.dtype([("i", "<u4"), ("j", "<u4"), ("pmi", "<f4")])


def load_vocab_counts_by_id(vocab_path: Path) -> np.ndarray:
    counts_by_id = {}
    max_id = -1

    with vocab_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"Bad vocab.tsv line {line_no}: {line!r}")

            _, sid, scount = parts
            wid = int(sid)
            cnt = int(scount)
            counts_by_id[wid] = cnt
            if wid > max_id:
                max_id = wid

    if max_id < 0:
        raise ValueError(f"No vocabulary entries found in {vocab_path}")

    counts = np.zeros(max_id + 1, dtype=np.float64)
    for wid, cnt in counts_by_id.items():
        counts[wid] = float(cnt)

    return counts


def list_bucket_files(cooc_dir: Path) -> List[Path]:
    files = sorted(cooc_dir.glob("b*.bin"))
    if not files:
        raise FileNotFoundError(f"No cooc bucket files found in {cooc_dir}")
    return files


def compute_pmi_values(
    i_arr: np.ndarray,
    j_arr: np.ndarray,
    c_arr: np.ndarray,
    counts: np.ndarray,
    total_tokens: float,
    ppmi: bool,
) -> np.ndarray:
    ci = counts[i_arr]
    cj = counts[j_arr]
    denom = ci * cj

    # Valid only when cooc count, word counts, and denominator are all positive.
    valid = (c_arr > 0.0) & (ci > 0.0) & (cj > 0.0) & (denom > 0.0)
    out = np.empty(c_arr.shape[0], dtype=np.float32)
    out.fill(np.nan)

    if valid.any():
        vals = np.log((c_arr[valid] * total_tokens) / denom[valid])
        if ppmi:
            vals = np.maximum(vals, 0.0)
        out[valid] = vals.astype(np.float32, copy=False)

    return out


def build_pmi_buckets(
    out_dir: Path,
    chunk_records: int,
    ppmi: bool,
    save_npz: bool,
) -> None:
    vocab_path = out_dir / "vocab.tsv"
    cooc_dir = out_dir / "cooc"

    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing {vocab_path}")
    if not cooc_dir.exists():
        raise FileNotFoundError(f"Missing {cooc_dir}")

    counts = load_vocab_counts_by_id(vocab_path)
    total_tokens = float(counts.sum())
    if total_tokens <= 0.0:
        raise ValueError("Total token count from vocab.tsv is not positive")

    bucket_files = list_bucket_files(cooc_dir)
    pmi_dir = out_dir / ("ppmi" if ppmi else "pmi")
    pmi_dir.mkdir(parents=True, exist_ok=True)

    all_i = []
    all_j = []
    all_v = []

    written = 0
    skipped_invalid = 0

    for bf in bucket_files:
        out_path = pmi_dir / bf.name
        with bf.open("rb") as fin, out_path.open("wb") as fout:
            while True:
                arr = np.fromfile(fin, dtype=COOC_REC_DTYPE, count=chunk_records)
                if arr.size == 0:
                    break

                i_arr = arr["i"]
                j_arr = arr["j"]
                c_arr = arr["c"].astype(np.float64, copy=False)

                pmi_vals = compute_pmi_values(
                    i_arr=i_arr,
                    j_arr=j_arr,
                    c_arr=c_arr,
                    counts=counts,
                    total_tokens=total_tokens,
                    ppmi=ppmi,
                )

                valid = np.isfinite(pmi_vals)
                if not valid.any():
                    skipped_invalid += arr.size
                    continue

                skipped_invalid += int((~valid).sum())
                i_valid = i_arr[valid]
                j_valid = j_arr[valid]
                p_valid = pmi_vals[valid]

                rec = np.empty(i_valid.shape[0], dtype=PMI_REC_DTYPE)
                rec["i"] = i_valid
                rec["j"] = j_valid
                rec["pmi"] = p_valid
                rec.tofile(fout)

                written += rec.shape[0]

                if save_npz:
                    all_i.append(i_valid.astype(np.int64, copy=False))
                    all_j.append(j_valid.astype(np.int64, copy=False))
                    all_v.append(p_valid.astype(np.float32, copy=False))

    print(f"Total vocab size: {counts.shape[0]:,}")
    print(f"Total tokens (sum of vocab counts): {int(total_tokens):,}")
    print(f"Wrote {written:,} PMI entries to {pmi_dir}")
    if skipped_invalid:
        print(f"Skipped {skipped_invalid:,} entries with non-positive denominator/count")

    if save_npz:
        if sparse is None:
            raise ImportError("scipy is required for --save_npz")

        if all_i:
            rows = np.concatenate(all_i)
            cols = np.concatenate(all_j)
            vals = np.concatenate(all_v)
            mat = sparse.coo_matrix((vals, (rows, cols)), shape=(counts.shape[0], counts.shape[0]), dtype=np.float32).tocsr()
        else:
            mat = sparse.csr_matrix((counts.shape[0], counts.shape[0]), dtype=np.float32)

        npz_path = out_dir / ("ppmi_matrix.npz" if ppmi else "pmi_matrix.npz")
        sparse.save_npz(npz_path, mat)
        print(f"Saved sparse matrix to {npz_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build PMI matrix from cooc buckets using: "
            "PMI(i,j)=log( cooc(i,j)*N / (count(i)*count(j)) ), where N=sum(vocab counts)."
        )
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory containing vocab.tsv and cooc/",
    )
    ap.add_argument(
        "--chunk_records",
        type=int,
        default=2_000_000,
        help="Number of cooc records to process per chunk",
    )
    ap.add_argument(
        "--ppmi",
        action="store_true",
        help="Clamp PMI at 0 (Positive PMI)",
    )
    ap.add_argument(
        "--save_npz",
        action="store_true",
        help="Also save a single sparse SciPy matrix (.npz)",
    )

    args = ap.parse_args()

    build_pmi_buckets(
        out_dir=Path(args.out_dir),
        chunk_records=args.chunk_records,
        ppmi=args.ppmi,
        save_npz=args.save_npz,
    )


if __name__ == "__main__":
    main()
