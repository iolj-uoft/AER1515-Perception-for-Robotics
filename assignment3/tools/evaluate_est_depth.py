#!/usr/bin/env python3
"""
Evaluate estimated dense depth maps against ground-truth for assignment3.

This script compares per-sample predicted depth maps (in a directory) against the
ground-truth depth maps and computes MAE, RMSE and AbsRel. It supports .npy files
and 16-bit PNGs where depth was stored as uint16 = round(depth*scale) (default scale=256).

Usage:
  python evaluate_est_depth.py --gt_dir data/train/gt_depth --est_dir data/train/est_depth

Outputs: prints per-sample and overall metrics and optionally writes a CSV.
"""
import argparse
import os
import sys
import numpy as np
import cv2
import csv


def load_depth(path, scale=256.0):
    """Load depth map from .npy or PNG. Returns float32 meters array or None if fail."""
    if not os.path.exists(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        try:
            arr = np.load(path)
            return arr.astype(np.float32)
        except Exception:
            return None
    else:
        # assume image (PNG/JPG). Read unchanged to preserve bitdepth
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        # if 16-bit, scale down
        if img.dtype == np.uint16:
            return img.astype(np.float32) / float(scale)
        # if 8-bit, return as float (no reliable scale)
        return img.astype(np.float32)


def compute_metrics_pair(gt, pred, valid_mask=None):
    """Compute MAE, RMSE, AbsRel between gt and pred arrays over valid_mask (bool array)."""
    if valid_mask is None:
        valid_mask = np.isfinite(gt) & (gt > 0)
    # exclude pixels where gt is invalid (<=0 or NaN)
    valid = valid_mask & np.isfinite(pred)
    if not np.any(valid):
        return None
    gt_v = gt[valid]
    pred_v = pred[valid]
    err = np.abs(pred_v - gt_v)
    mae = float(np.mean(err))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    absrel = float(np.mean(err / gt_v))
    return mae, rmse, absrel, int(valid.sum())


def find_samples(gt_dir, est_dir):
    """Find sample names that exist in both dirs (by basename without extension)."""
    gt_files = [f for f in os.listdir(gt_dir) if not f.startswith('.')]
    est_files = [f for f in os.listdir(est_dir) if not f.startswith('.')]
    gt_names = set(os.path.splitext(f)[0] for f in gt_files)
    est_names = set(os.path.splitext(f)[0] for f in est_files)
    common = sorted(list(gt_names & est_names))
    return common


def main():
    parser = argparse.ArgumentParser(description='Evaluate estimated depth maps against GT')
    parser.add_argument('--gt_dir', required=False, help='Directory with ground-truth depth maps', default="data/train/gt_depth")
    parser.add_argument('--est_dir', required=False, help='Directory with estimated depth maps', default="data/train/est_depth")
    parser.add_argument('--scale', type=float, default=256.0, help='Scale used for uint16 PNG -> meters (default 256)')
    parser.add_argument('--out_csv', default=None, help='Optional CSV path to write per-sample metrics')
    args = parser.parse_args()

    gt_dir = args.gt_dir
    est_dir = args.est_dir
    if not os.path.isdir(gt_dir):
        print(f"GT directory not found: {gt_dir}")
        sys.exit(1)
    if not os.path.isdir(est_dir):
        print(f"Est directory not found: {est_dir}")
        sys.exit(1)

    samples = find_samples(gt_dir, est_dir)
    if not samples:
        print("No common samples found between GT and Est directories.")
        sys.exit(1)

    all_metrics = []
    total_valid = 0
    weighted_mae = 0.0
    weighted_rmse_sq_sum = 0.0

    for s in samples:
        gt_path = None
        est_path = None
        # choose file with extension: prefer .npy then .png
        for ext in ['.npy', '.npz', '.png', '.PNG', '.jpg', '.jpeg']:
            p = os.path.join(gt_dir, s + ext)
            if os.path.exists(p):
                gt_path = p
                break
        for ext in ['.npy', '.npz', '.png', '.PNG', '.jpg', '.jpeg']:
            p = os.path.join(est_dir, s + ext)
            if os.path.exists(p):
                est_path = p
                break
        if gt_path is None or est_path is None:
            # skip if one missing
            continue

        gt = load_depth(gt_path, scale=args.scale)
        pred = load_depth(est_path, scale=args.scale)
        if gt is None or pred is None:
            print(f"Failed to load {s}: gt={gt_path}, pred={est_path}")
            continue

        # if shapes differ, try to resize predicted to gt resolution
        if gt.shape != pred.shape:
            # resize pred to gt shape using linear interpolation
            pred_resized = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
            pred = pred_resized

        res = compute_metrics_pair(gt, pred)
        if res is None:
            print(f"No valid pixels for sample {s}; skipping")
            continue
        mae, rmse, absrel, n_valid = res
        all_metrics.append((s, mae, rmse, absrel, n_valid))
        print(f"{s}: n={n_valid} MAE={mae:.4f} RMSE={rmse:.4f} AbsRel={absrel:.4f}")

        # accumulate weighted stats
        total_valid += n_valid
        weighted_mae += mae * n_valid
        weighted_rmse_sq_sum += (rmse ** 2) * n_valid

    if not all_metrics:
        print("No valid comparisons found.")
        sys.exit(1)

    overall_mae = weighted_mae / total_valid
    overall_rmse = float(np.sqrt(weighted_rmse_sq_sum / total_valid))
    overall_absrel = float(np.mean([m[3] for m in all_metrics]))

    print('\nOverall:')
    print(f"  Total valid pixels: {total_valid}")
    print(f"  MAE: {overall_mae:.4f} m")
    print(f"  RMSE: {overall_rmse:.4f} m")
    print(f"  AbsRel (mean of samples): {overall_absrel:.4f}")

    if args.out_csv:
        with open(args.out_csv, 'w', newline='') as cf:
            writer = csv.writer(cf)
            writer.writerow(['sample', 'n_valid', 'mae_m', 'rmse_m', 'absrel'])
            for s, mae, rmse, absrel, n in all_metrics:
                writer.writerow([s, n, f"{mae:.6f}", f"{rmse:.6f}", f"{absrel:.6f}"])
        print(f"Wrote per-sample metrics to {args.out_csv}")


if __name__ == '__main__':
    main()
