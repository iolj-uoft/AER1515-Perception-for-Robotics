import os
import glob
import cv2 as cv
import numpy as np

def _resize_to_width(img, target_w):
    h, w = img.shape[:2]
    if w == target_w:
        return img
    scale = target_w / float(w)
    new_h = int(round(h * scale))
    return cv.resize(img, (target_w, new_h), interpolation=cv.INTER_AREA)

def combine_plot_suffix(suffix: str, fig_dir: str = "figures", out_name: str = None, spacing: int = 20):
    """
    Find all files matching "*_{suffix}.png" in fig_dir, vertically stack them
    (in filename-sorted order) with white spacing between, and save as out_name.
    """
    pattern = os.path.join(fig_dir, f"*_{suffix}.png")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files found for suffix '{suffix}' in {fig_dir}")
        return None

    imgs = []
    widths = []
    for f in files:
        img = cv.imread(f, cv.IMREAD_COLOR)
        if img is None:
            print(f"Warning: failed to read {f}, skipping")
            continue
        imgs.append(img)
        widths.append(img.shape[1])

    if not imgs:
        print(f"No readable images for suffix '{suffix}'")
        return None

    max_w = max(widths)
    resized = []
    white = None
    for img in imgs:
        r = _resize_to_width(img, max_w)
        resized.append(r)
    white = 255 * np.ones((spacing, max_w, 3), dtype=np.uint8)

    stacked_parts = []
    for i, r in enumerate(resized):
        stacked_parts.append(r)
        if i < len(resized) - 1:
            stacked_parts.append(white)

    combined = np.vstack(stacked_parts)

    if out_name is None:
        out_name = f"combined_{suffix}.png"
    out_path = os.path.join(fig_dir, out_name)
    cv.imwrite(out_path, combined)
    print(f"Saved combined image: {out_path}")
    return out_path

def combine_all_plots(fig_dir: str = "figures"):
    """
    Combine common plot types across objects into stacked images.
    Adjust the suffix list if you have other filenames to combine.
    """
    suffixes = [
        "before_registration",
        "after_registration",
        "icp_mean_distance",
        "icp_translation_components"
    ]
    for s in suffixes:
        combine_plot_suffix(s, fig_dir=fig_dir, out_name=f"combined_{s}.png", spacing=20)


if __name__ == "__main__":
    # call combiner after main to produce stacked images
    FIG_DIR = "figures"
    os.makedirs(FIG_DIR, exist_ok=True)
    combine_all_plots(fig_dir=FIG_DIR)