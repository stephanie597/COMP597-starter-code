#!/usr/bin/env python
import argparse
import json
import multiprocessing
import os
from collections import defaultdict
from pathlib import Path
import time
import random

#import torchcompat.core as acc
import torch
from tqdm import tqdm


def write(args):
    import torchvision.transforms as transforms

    offset, outdir, prefix, size = args

    seed = int(time.time() + offset)

    torch.manual_seed(seed)
    random.seed(seed)

    img = torch.randint(0, 256, size, dtype=torch.uint8)
    img = transforms.ToPILImage()(img)
    target = offset % 1000
    class_val = int(target)

    if not prefix:  # train
        image_name = f"{class_val}_{offset}"
    else:  # val, test
        image_name = f"{prefix}{int(offset):08d}"

    path = os.path.join(outdir, str(class_val))
    os.makedirs(path, exist_ok=True)

    image_path = os.path.join(path, f"{image_name}.JPEG")
    img.save(image_path)


def generate(image_size, n, outdir, prefix="", start=0):
    work_items = []
    for i in range(n):
        work_items.append([start + i, outdir, prefix, image_size])

    n_worker = min(multiprocessing.cpu_count(), 8)
    with multiprocessing.Pool(n_worker) as pool:
        for _ in tqdm(pool.imap_unordered(write, work_items), total=n):
            pass


def count_images(path):
    count = defaultdict(int)
    for root, _, files in tqdm(os.walk(path)):
        # expected: .../<split>/<class_id>/
        parts = root.split("/")
        if len(parts) >= 2:
            split = parts[-2]
            count[split] += len(files)
    return count


def dir_size_bytes(path: Path) -> int:
    """Return total size in bytes of all files under path."""
    total = 0
    if not path.exists():
        return 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def generate_sets(root, sets, shape, max_size_gb=None, chunk_size=2000):
    """
    Ensure dataset has desired image counts, BUT stop early if max_size_gb is reached.
    """
    root = Path(root)
    sentinel = root / "done"
    os.makedirs(root, exist_ok=True)

    max_bytes = None
    if max_size_gb is not None:
        max_bytes = int(float(max_size_gb) * 1024**3)

    total_images = count_images(root)

    def reached_cap() -> bool:
        if max_bytes is None:
            return False
        current = dir_size_bytes(root)
        return current >= max_bytes

    for split, target_count in sets.items():
        current_count = total_images.get(split, 0)

        if split == "train":
            prefix = ""
        else:
            prefix = f"ILSVRC2012_{split}_"

        if current_count >= target_count:
            continue

        print(f"Generating {split} (current {current_count}) (target: {target_count})")

        # Generate in chunks, checking disk size in between to enforce max_size_gb.
        split_dir = os.path.join(root, split)
        while current_count < target_count:
            if reached_cap():
                cur_gb = dir_size_bytes(root) / 1024**3
                print(f"[STOP] Reached size cap: {cur_gb:.2f} GB (limit {max_size_gb} GB).")
                break

            remaining = target_count - current_count
            this_n = min(chunk_size, remaining)

            generate(
                shape,
                this_n,
                split_dir,
                prefix=prefix,
                start=current_count,
            )

            # refresh counts for this split only (fast path)
            # (we could recount only split_dir, but simplest is recount all once per chunk)
            total_images = count_images(root)
            current_count = total_images.get(split, 0)

        # If we hit cap in this split, don't attempt other splits either.
        if reached_cap():
            break

    # Write what we *intended* and what we *actually* have.
    final_counts = count_images(root)
    meta = {
        "requested": sets,
        "actual": dict(final_counts),
        "image_shape": list(shape),
        "max_size_gb": max_size_gb,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(sentinel, "w") as fp:
        json.dump(meta, fp, indent=2)


def device_count():
    return torch.cuda.device_count() if torch.cuda.is_available() else 1


def fakeimagenet_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--batch-count", default=60, type=int)
    parser.add_argument("--device-count", default=device_count(), type=int)
    parser.add_argument("--device", default=None, type=str)
    parser.add_argument("--image-size", default=[3, 384, 384], type=int, nargs="+")
    parser.add_argument("--val", default=0.1, type=float)
    parser.add_argument("--test", default=0.1, type=float)
    parser.add_argument("--output", default=os.getenv("MILABENCH_DIR_DATA", None), type=str)

    # NEW: size cap
    parser.add_argument(
        "--max-size-gb",
        default=None,
        type=float,
        help="Stop generating once FakeImageNet directory reaches this many GB (approx, based on actual file sizes).",
    )

    # (optional) chunk tuning
    parser.add_argument(
        "--chunk-size",
        default=2000,
        type=int,
        help="How many images to generate per chunk before re-checking max-size-gb.",
    )

    args, _ = parser.parse_known_args()
    return args


def generate_fakeimagenet(args=None):
    if args is None:
        args = fakeimagenet_args()

    if overrides := os.getenv("MILABENCH_TESTING_PREPARE"):
        bs, bc = overrides.split(",")
        args.batch_size, args.batch_count = int(bs), int(bc)

    assert args.output is not None, "Output directory is required"
    data_directory = args.output

    dest = os.path.join(data_directory, "FakeImageNet")
    print(f"Generating fake data into {dest}...")

    total_images = args.batch_size * args.batch_count * args.device_count
    size_spec = {
        "train": int(total_images),
        "val": int(total_images * args.val),
        "test": int(total_images * args.test),
    }

    generate_sets(
        dest,
        size_spec,
        args.image_size,
        max_size_gb=args.max_size_gb,
        chunk_size=args.chunk_size,
    )

    # labels.txt based on what exists (may be partial if stopped early)
    labels = set([int(entry.name) for entry in Path(dest).glob("*/*/")])
    with open(os.path.join(dest, "labels.txt"), "wt") as _f:
        _f.writelines([f"{l},{l}\n" for l in sorted(labels)])

    print("Done!")


if __name__ == "__main__":
    args = fakeimagenet_args()
    generate_fakeimagenet(args)