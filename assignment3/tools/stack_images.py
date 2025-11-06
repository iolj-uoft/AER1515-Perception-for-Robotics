import os
import argparse
import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt 

def main():
    parser = argparse.ArgumentParser(description="Create subplots of all images in a directory.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing images to plot")
    parser.add_argument("--output_path", type=str, default="figures/stacked_images.png", help="Output path for the subplot image")
    parser.add_argument("--extension", type=str, default="png", help="Image extension to look for (default: png)")
    args = parser.parse_args()
    
    # Find all images
    pattern = os.path.join(args.input_dir, f"*.{args.extension}")
    image_paths = sorted(glob(pattern))
    
    if not image_paths:
        print(f"No {args.extension} images found in {args.input_dir}")
        return
    
    # Load images
    images = []
    valid_paths = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"Warning: Could not load {path}")
            continue
        images.append(img)
        valid_paths.append(path)
    
    if not images:
        print("No valid images to plot")
        return
    
    # Create subplots (vertical stack)
    n = len(images)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(8, 3 * n))
    if n == 1:
        axes = [axes]  # Ensure axes is a list
    
    for i, (img, path) in enumerate(zip(images, valid_paths)):
        # Convert BGR to RGB for matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img_rgb)
        axes[i].set_title(os.path.basename(path))
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    plt.savefig(args.output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved subplot image to {args.output_path}")

if __name__ == '__main__':
    main()