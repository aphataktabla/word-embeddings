import os
import argparse
import zstandard as zstd
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--max_docs", type=int, default=40_000_000)
    parser.add_argument("--docs_per_shard", type=int, default=1_000_000)
    parser.add_argument("--compression_level", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading CC100 English (streaming)...")
    ds = load_dataset(
        "cc100",
        "en",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    compressor = zstd.ZstdCompressor(level=args.compression_level)

    shard_id = 0
    doc_count = 0
    shard_doc_count = 0

    current_file = None
    current_writer = None
    current_doc_parts = []

    def open_new_shard(shard_id):
        filename = os.path.join(args.out_dir, f"shard_{shard_id:05d}.txt.zst")
        print(f"Opening shard {filename}")
        f = open(filename, "wb")
        writer = compressor.stream_writer(f)
        return f, writer

    def flush_current_doc():
        nonlocal shard_id, doc_count, shard_doc_count
        nonlocal current_file, current_writer, current_doc_parts

        if not current_doc_parts or doc_count >= args.max_docs:
            current_doc_parts = []
            return

        if shard_doc_count == 0:
            current_file, current_writer = open_new_shard(shard_id)

        text = " ".join(current_doc_parts).strip()
        current_doc_parts = []
        if not text:
            return

        current_writer.write((text + "\n").encode("utf-8"))

        doc_count += 1
        shard_doc_count += 1

        if doc_count % 100000 == 0:
            print(f"Exported {doc_count:,} documents...")

        if shard_doc_count >= args.docs_per_shard:
            current_writer.close()
            current_file.close()
            current_writer = None
            current_file = None
            shard_id += 1
            shard_doc_count = 0

    for example in ds:
        if doc_count >= args.max_docs:
            break

        text = example["text"].strip().replace("\n", " ")
        if not text:
            flush_current_doc()
            continue

        current_doc_parts.append(text)

    if doc_count < args.max_docs:
        flush_current_doc()

    if current_writer:
        current_writer.close()
    if current_file:
        current_file.close()

    print(f"\nDone. Exported {doc_count:,} documents into {shard_id + 1} shards.")

if __name__ == "__main__":
    main()
