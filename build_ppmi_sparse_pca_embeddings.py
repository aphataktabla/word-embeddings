#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
from scipy import sparse
from sklearn.decomposition import MiniBatchSparsePCA, SparsePCA


PPMI_REC_DTYPE = np.dtype([("i", "<u4"), ("j", "<u4"), ("pmi", "<f4")])


def load_vocab(vocab_path: Path) -> tuple[int, Dict[str, int]]:
    max_id = -1
    token_to_id: Dict[str, int] = {}
    with vocab_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"Bad vocab.tsv line {line_no}: {line!r}")
            token = parts[0]
            wid = int(parts[1])
            token_to_id[token] = wid
            if wid > max_id:
                max_id = wid

    if max_id < 0:
        raise ValueError(f"No vocabulary entries found in {vocab_path}")
    return max_id + 1, token_to_id


def list_bucket_files(ppmi_dir: Path) -> List[Path]:
    files = sorted(ppmi_dir.glob("b*.bin"))
    if not files:
        raise FileNotFoundError(f"No PPMI bucket files found in {ppmi_dir}")
    return files


def parse_bats_token(raw: str) -> List[str]:
    return [part.strip() for part in raw.split("/") if part.strip()]


def load_priority_ids_from_bats(
    bats_dir: Path,
    token_to_id: Dict[str, int],
) -> List[int]:
    priority_ids: Set[int] = set()
    for path in sorted(bats_dir.glob("*.txt")):
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                lhs_tokens = parse_bats_token(parts[0])
                rhs_tokens = parse_bats_token(parts[1])
                for token in lhs_tokens + rhs_tokens:
                    if token in token_to_id:
                        priority_ids.add(token_to_id[token])
    return sorted(priority_ids)


def choose_vocab_ids(
    vocab_size: int,
    max_vocab: int,
    priority_ids: List[int],
) -> np.ndarray:
    cutoff = min(vocab_size, max_vocab)
    selected: List[int] = []
    seen: Set[int] = set()

    for wid in priority_ids:
        if 0 <= wid < vocab_size and wid not in seen:
            selected.append(wid)
            seen.add(wid)
            if len(selected) >= cutoff:
                return np.array(selected, dtype=np.uint32)

    for wid in range(vocab_size):
        if wid not in seen:
            selected.append(wid)
            if len(selected) >= cutoff:
                break

    return np.array(selected, dtype=np.uint32)


def load_ppmi_submatrix(
    ppmi_dir: Path,
    selected_ids: np.ndarray,
    chunk_records: int,
) -> sparse.csr_matrix:
    cutoff = int(selected_ids.shape[0])
    id_map = np.full(int(selected_ids.max()) + 1, -1, dtype=np.int32)
    id_map[selected_ids] = np.arange(cutoff, dtype=np.int32)

    rows = []
    cols = []
    vals = []

    for bf in list_bucket_files(ppmi_dir):
        with bf.open("rb") as f:
            while True:
                arr = np.fromfile(f, dtype=PPMI_REC_DTYPE, count=chunk_records)
                if arr.size == 0:
                    break

                mask_i = arr["i"] < id_map.shape[0]
                mask_j = arr["j"] < id_map.shape[0]
                mapped_i = np.full(arr.shape[0], -1, dtype=np.int32)
                mapped_j = np.full(arr.shape[0], -1, dtype=np.int32)
                mapped_i[mask_i] = id_map[arr["i"][mask_i]]
                mapped_j[mask_j] = id_map[arr["j"][mask_j]]
                mask = (mapped_i >= 0) & (mapped_j >= 0)
                if not mask.any():
                    continue

                rows.append(mapped_i[mask].astype(np.int64, copy=False))
                cols.append(mapped_j[mask].astype(np.int64, copy=False))
                vals.append(arr["pmi"][mask].astype(np.float32, copy=False))

    if not rows:
        return sparse.csr_matrix((cutoff, cutoff), dtype=np.float32)

    row_idx = np.concatenate(rows)
    col_idx = np.concatenate(cols)
    data = np.concatenate(vals)
    return sparse.coo_matrix(
        (data, (row_idx, col_idx)),
        shape=(cutoff, cutoff),
        dtype=np.float32,
    ).tocsr()


def run_sparse_pca(
    mat: sparse.csr_matrix,
    n_components: int,
    alpha: float,
    ridge_alpha: float,
    random_state: int,
    max_iter: int,
    method: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = mat.toarray()

    if method == "minibatch":
        model = MiniBatchSparsePCA(
            n_components=n_components,
            alpha=alpha,
            ridge_alpha=ridge_alpha,
            batch_size=batch_size,
            random_state=random_state,
            max_iter=max_iter,
            verbose=1,
        )
    else:
        model = SparsePCA(
            n_components=n_components,
            alpha=alpha,
            ridge_alpha=ridge_alpha,
            random_state=random_state,
            max_iter=max_iter,
            verbose=1,
        )

    embeddings = model.fit_transform(x).astype(np.float32, copy=False)
    components = model.components_.astype(np.float32, copy=False)
    return embeddings, components


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run sparse PCA on a restricted submatrix of the PPMI matrix. "
            "BATS encyclopedic vocab ids are prioritized in the selected subset."
        )
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory containing vocab.tsv and ppmi/",
    )
    ap.add_argument(
        "--max_vocab",
        type=int,
        default=10000,
        help="Number of vocab ids to keep in the sparse PCA submatrix",
    )
    ap.add_argument(
        "--n_components",
        type=int,
        default=300,
        help="Number of sparse PCA components",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Sparsity controlling parameter for SparsePCA",
    )
    ap.add_argument(
        "--ridge_alpha",
        type=float,
        default=0.01,
        help="Ridge shrinkage used when transforming data",
    )
    ap.add_argument(
        "--max_iter",
        type=int,
        default=100,
        help="Maximum number of sparse PCA iterations",
    )
    ap.add_argument(
        "--chunk_records",
        type=int,
        default=2_000_000,
        help="Records to read per chunk when loading bucketed PPMI data",
    )
    ap.add_argument(
        "--method",
        choices=["standard", "minibatch"],
        default="minibatch",
        help="Sparse PCA solver to use",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Mini-batch size when using --method minibatch",
    )
    ap.add_argument(
        "--random_state",
        type=int,
        default=0,
        help="Random seed",
    )
    ap.add_argument(
        "--bats_encyclopedic_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "BATS_3.0" / "3_Encyclopedic_semantics"),
        help="Directory of BATS encyclopedic files whose vocab ids should be prioritized",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    vocab_path = out_dir / "vocab.tsv"
    ppmi_dir = out_dir / "ppmi"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing {vocab_path}")
    if not ppmi_dir.exists():
        raise FileNotFoundError(f"Missing {ppmi_dir}")

    vocab_size, token_to_id = load_vocab(vocab_path)
    if args.max_vocab <= 1:
        raise ValueError("--max_vocab must be > 1")

    cutoff = min(vocab_size, args.max_vocab)
    if args.n_components <= 0 or args.n_components > cutoff:
        raise ValueError(f"--n_components must be in [1, {cutoff}]")

    priority_ids = load_priority_ids_from_bats(
        bats_dir=Path(args.bats_encyclopedic_dir),
        token_to_id=token_to_id,
    )
    selected_ids = choose_vocab_ids(
        vocab_size=vocab_size,
        max_vocab=args.max_vocab,
        priority_ids=priority_ids,
    )

    mat = load_ppmi_submatrix(
        ppmi_dir=ppmi_dir,
        selected_ids=selected_ids,
        chunk_records=args.chunk_records,
    )

    embeddings, components = run_sparse_pca(
        mat=mat,
        n_components=args.n_components,
        alpha=args.alpha,
        ridge_alpha=args.ridge_alpha,
        random_state=args.random_state,
        max_iter=args.max_iter,
        method=args.method,
        batch_size=args.batch_size,
    )

    stem = f"ppmi_sparse_pca_v{cutoff}_k{args.n_components}"
    emb_path = out_dir / f"{stem}_embeddings.npy"
    comp_path = out_dir / f"{stem}_components.npy"
    ids_path = out_dir / f"{stem}_vocab_ids.npy"

    np.save(emb_path, embeddings)
    np.save(comp_path, components)
    np.save(ids_path, selected_ids)

    print(f"Loaded submatrix shape: {mat.shape}")
    print(f"Prioritized {min(len(priority_ids), cutoff)} BATS encyclopedic vocab ids")
    print(f"Wrote sparse PCA embeddings to {emb_path}")
    print(f"Wrote sparse PCA components to {comp_path}")
    print(f"Wrote selected vocab ids to {ids_path}")


if __name__ == "__main__":
    main()
