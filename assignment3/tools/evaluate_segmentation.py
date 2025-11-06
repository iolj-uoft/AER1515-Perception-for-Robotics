#!/usr/bin/env python3
import os
import argparse
from glob import glob

import numpy as np
import cv2


def binarize_masks(est, gt):
    """
    Convert raw mask images to binary {0,1} where 1 = foreground (car), 0 = background.

    Assignment convention:
      - Estimated mask: car = 0, background = 255
      - Ground-truth mask: car = any value < 255, background = 255
    """
    est_bin = (est == 0).astype(np.uint8)
    gt_bin = (gt != 255).astype(np.uint8)
    return est_bin, gt_bin


def ensure_same_size(est, gt):
    if est.shape != gt.shape:
        est = cv2.resize(est, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    return est


def compute_confusion(pred, gt):
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    tp = np.logical_and(pred_b, gt_b).sum()
    fp = np.logical_and(pred_b, np.logical_not(gt_b)).sum()
    fn = np.logical_and(np.logical_not(pred_b), gt_b).sum()
    tn = np.logical_and(np.logical_not(pred_b), np.logical_not(gt_b)).sum()
    return int(tp), int(fp), int(fn), int(tn)


def safe_div(num, den):
    return num / den if den != 0 else 0.0


def compute_metrics(pred, gt):
    tp, fp, fn, tn = compute_confusion(pred, gt)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, (precision + recall)) if (precision + recall) > 0 else 0.0
    union = tp + fp + fn
    iou = safe_div(tp, union)
    acc = safe_div(tp + tn, tp + fp + fn + tn)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": acc,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Could not load: {path}")
    return img


def pair_files(est_dir, gt_dir, pattern="*.png"):
    est_paths = {os.path.splitext(os.path.basename(p))[0]: p for p in glob(os.path.join(est_dir, pattern))}
    gt_paths = {os.path.splitext(os.path.basename(p))[0]: p for p in glob(os.path.join(gt_dir, pattern))}
    common = sorted(set(est_paths.keys()) & set(gt_paths.keys()))
    missing_est = sorted(set(gt_paths.keys()) - set(est_paths.keys()))
    missing_gt = sorted(set(est_paths.keys()) - set(gt_paths.keys()))
    if missing_est:
        print(f"Warning: {len(missing_est)} GT images have no matching EST: {missing_est[:5]}{' ...' if len(missing_est) > 5 else ''}")
    if missing_gt:
        print(f"Warning: {len(missing_gt)} EST images have no matching GT: {missing_gt[:5]}{' ...' if len(missing_gt) > 5 else ''}")
    return [(k, est_paths[k], gt_paths[k]) for k in common]


def evaluate_sample(est_path, gt_path, print_debug=False):
    est_raw = load_gray(est_path)
    gt_raw = load_gray(gt_path)

    est_raw = ensure_same_size(est_raw, gt_raw)
    est_bin, gt_bin = binarize_masks(est_raw, gt_raw)

    if print_debug:
        est_fg = int(est_bin.sum())
        gt_fg = int(gt_bin.sum())
        total = est_bin.size
        print(f"Debug {os.path.basename(est_path)}: Est FG={est_fg}/{total} ({est_fg/total:.2%}), GT FG={gt_fg}/{total} ({gt_fg/total:.2%})")

    return compute_metrics(est_bin, gt_bin)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation masks against ground truth using precision/recall (AER1515 grading metric)."
    )
    parser.add_argument("--est_dir", type=str, default="data/train/est_segmentation",
                        help="Directory with estimated masks (PNG). Car=0, Background=255.")
    parser.add_argument("--gt_dir", type=str, default="data/train/gt_segmentation",
                        help="Directory with ground-truth masks (PNG). Car<255, Background=255.")
    parser.add_argument("--out_csv", type=str, default=None, help="Optional: save per-sample results to CSV.")
    parser.add_argument("--debug", action="store_true", help="Print foreground pixel ratios for sanity checks.")
    args = parser.parse_args()

    pairs = pair_files(args.est_dir, args.gt_dir)
    if not pairs:
        print("No matching files found. Check your directories.")
        return

    results = []
    sum_tp = sum_fp = sum_fn = sum_tn = 0

    for name, est_path, gt_path in pairs:
        try:
            res = evaluate_sample(est_path, gt_path, print_debug=args.debug)
        except Exception as e:
            print(f"Warning: skipping {name} due to {e}")
            continue

        results.append({"sample": name, **res})
        sum_tp += res["tp"]
        sum_fp += res["fp"]
        sum_fn += res["fn"]
        sum_tn += res["tn"]

        print(f"{name}: Prec={res['precision']:.4f}, Rec={res['recall']:.4f}, "
              f"F1={res['f1']:.4f}, IoU={res['iou']:.4f}, Acc={res['accuracy']:.4f}")

    if not results:
        print("No valid samples evaluated.")
        return

    # Macro averages (mean of per-sample)
    macro_precision = float(np.mean([r["precision"] for r in results]))
    macro_recall = float(np.mean([r["recall"] for r in results]))
    macro_f1 = float(np.mean([r["f1"] for r in results]))
    macro_iou = float(np.mean([r["iou"] for r in results]))
    macro_acc = float(np.mean([r["accuracy"] for r in results]))

    # Micro averages (sum over all pixels)
    micro_precision = safe_div(sum_tp, (sum_tp + sum_fp))
    micro_recall = safe_div(sum_tp, (sum_tp + sum_fn))
    micro_f1 = safe_div(2 * micro_precision * micro_recall, (micro_precision + micro_recall)) if (micro_precision + micro_recall) > 0 else 0.0
    micro_iou = safe_div(sum_tp, (sum_tp + sum_fp + sum_fn))
    micro_acc = safe_div(sum_tp + sum_tn, sum_tp + sum_fp + sum_fn + sum_tn)

    print("\n=== Macro (average per image) ===")
    print(f"Precision={macro_precision:.4f}, Recall={macro_recall:.4f}, F1={macro_f1:.4f}, IoU={macro_iou:.4f}, Acc={macro_acc:.4f}")

    print("=== Micro (global over all pixels) ===")
    print(f"Precision={micro_precision:.4f}, Recall={micro_recall:.4f}, F1={micro_f1:.4f}, IoU={micro_iou:.4f}, Acc={micro_acc:.4f}")

    if args.out_csv:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(args.out_csv, index=False)
        print(f"Saved detailed results to {args.out_csv}")


if __name__ == "__main__":
    main()
