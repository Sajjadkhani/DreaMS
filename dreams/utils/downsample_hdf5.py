#!/usr/bin/env python3
"""
downsample_hdf5.py  –  shrink an HDF5 file to a given fraction

USAGE
  python downsample_hdf5.py GeMS_A1_DreaMS_rand50k.hdf5 \
      --fraction 0.10 --seed 42 \
      --output GeMS_A1_DreaMS_rand50k_10pct.hdf5
"""
import argparse
import os
import numpy as np
import h5py
import random
import sys


def copy_dataset(src_ds, dst_parent, name, idx=None):
    "Copy a dataset (optionally only the rows in idx) with compression + attrs."
    data = src_ds[idx] if idx is not None else src_ds[...]
    # gzip-9 + shuffle, automatic chunking
    dst_ds = dst_parent.create_dataset(
        name, data=data,
        compression="gzip", compression_opts=9, shuffle=True, chunks=True
    )
    for k, v in src_ds.attrs.items():
        dst_ds.attrs[k] = v


def recursive_copy(src_grp, dst_grp, fraction, rng):
    "Walk the tree; down-sample row-like datasets."
    for name, item in src_grp.items():
        if isinstance(item, h5py.Dataset):
            # Heuristic: if the first dimension looks like a ‘row’ axis → sample
            if item.shape and item.shape[0] > 10:
                n = item.shape[0]
                m = max(1, int(round(n * fraction)))
                idx = rng.choice(n, size=m, replace=False)
                idx.sort()
                copy_dataset(item, dst_grp, name, idx)
            else:
                # copy whole small/scalar
                copy_dataset(item, dst_grp, name)
        else:                                             # it’s a Group
            new_grp = dst_grp.create_group(name)
            for k, v in item.attrs.items():               # copy group attributes
                new_grp.attrs[k] = v
            recursive_copy(item, new_grp, fraction, rng)  # recurse


def main(src, out, fraction, seed):
    rng = np.random.default_rng(seed)
    with h5py.File(src, "r") as fin, h5py.File(out, "w") as fout:
        # copy root attributes (if any)
        for k, v in fin.attrs.items():
            fout.attrs[k] = v
        recursive_copy(fin, fout, fraction, rng)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Down-sample an HDF5 file")
    p.add_argument("src_file")
    p.add_argument("--fraction", "-f", type=float, default=0.0010,
                   help="keep this fraction of rows (default 0.10)")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for reproducibility")
    p.add_argument("--output", "-o", help="output file name")
    args = p.parse_args()

    out_file = args.output or (
        os.path.splitext(args.src_file)[0]
        + f"_{int(args.fraction*100)}pct.hdf5"
    )
    main(args.src_file, out_file, args.fraction, args.seed)
    print(f"✓ Wrote {out_file}")
