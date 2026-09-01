"""Сверка построенного X6 с эталонным файлом: все колонки, все якоря.

Нужна, только если эталонный X6.npy с этапа разработки ещё лежит на диске.
Похожий X6 дал бы похожий nd_plain, похожий базис и тихо другие числа Stage C,
поэтому важно отличать «тот же объект» от «примерно такой же».

    py -3.11 src/s04_verify_x6.py --ref ПУТЬ_К_ЭТАЛОНУ
"""
import argparse
import json
import os

import numpy as np

import paths

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True, help="путь к эталонному X6.npy")
ap.add_argument("--ref-meta", default=None,
                help="meta6.json рядом с эталоном (по умолчанию - соседний файл)")
ap.add_argument("--tol", type=float, default=1e-5)
args = ap.parse_args()
if args.ref_meta is None:
    args.ref_meta = os.path.join(os.path.dirname(args.ref), "meta6.json")

mine = np.load("features/X6.npy", mmap_mode="r")
ref = np.load(args.ref, mmap_mode="r")
if mine.shape != ref.shape:
    raise SystemExit(f"shape {mine.shape} vs reference {ref.shape}")

names = json.load(open("features/meta6.json"))["names"]
rnames = json.load(open(args.ref_meta))["names"]
if names != rnames:
    bad = [(i, a, b) for i, (a, b) in enumerate(zip(names, rnames)) if a != b]
    raise SystemExit(f"column names differ at {bad[:5]}")
print(f"shape {mine.shape}, column names identical")

n = paths.dataset()[0]
K = mine.shape[0] // n
worst = np.zeros(79)
worst_blk = np.zeros(79, dtype=int)
scale = np.zeros(79)
for b in range(K):
    a = np.asarray(mine[b * n:(b + 1) * n], dtype=np.float64)
    r = np.asarray(ref[b * n:(b + 1) * n], dtype=np.float64)
    d = np.abs(a - r).max(0)
    s = np.abs(r).max(0)
    upd = d > worst
    worst[upd] = d[upd]
    worst_blk[upd] = b
    scale = np.maximum(scale, s)
    print(f"  block {b:2d}  max abs diff {d.max():.3e}", flush=True)

rel = worst / np.maximum(scale, 1e-12)
order = np.argsort(-rel)
print(f"\n{'column':26s} {'max abs diff':>13s} {'relative':>11s}  worst block")
for j in order[:12]:
    print(f"{names[j]:26s} {worst[j]:13.3e} {rel[j]:11.2e}  {worst_blk[j]}")

bad = [names[j] for j in range(79) if rel[j] > args.tol]
print(f"\ncolumns within {args.tol:g} relative: {79 - len(bad)}/79")
if bad:
    print(f"NOT matching: {bad}")
    raise SystemExit(1)
print("каждая колонка каждого якоря совпадает с эталоном до точности float32")
