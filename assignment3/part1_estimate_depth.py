import os
import sys

import cv2
import numpy as np
import kitti_dataHandler
import matplotlib.pyplot as plt

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["train", "test"], default="train", required=True, type=str)
    args = parser.parse_args()
    
    ################
    # Options
    ################
    # Input dir and output dir
    if (args.dataset == "train"):
        disp_dir = 'data/train/disparity'
        output_dir = 'data/train/est_depth'
        calib_dir = 'data/train/calib'
        sample_list = ['000001', '000002', '000003', '000004','000005', '000006', '000007', '000008', '000009', '000010']
    else:
        disp_dir = 'data/test/disparity'
        output_dir = 'data/test/est_depth'
        calib_dir = 'data/test/calib'
        sample_list = ['000011', '000012', '000013', '000014', '000015']
        test_images = []
        test_names = []
    ################

    for sample_name in (sample_list):
        # Read disparity map
        img_path = os.path.join(disp_dir, sample_name + ".png")
        disp = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
        disp /= 256.0

        # Read calibration info
        calib_path = os.path.join(calib_dir, sample_name + ".txt")
        frame_calib = kitti_dataHandler.read_frame_calib(calib_path)
        stereo_calib = kitti_dataHandler.get_stereo_calibration(frame_calib.p2, frame_calib.p3)
        
        # Calculate depth (z = f*B/disp)
        os.makedirs(output_dir, exist_ok=True)        
        f = stereo_calib.f
        baseline = stereo_calib.baseline
        with np.errstate(divide='ignore', invalid='ignore'):
            depth = np.zeros_like(disp, dtype=np.float32)
            valid = disp > 0
            depth[valid] = (f * baseline) / disp[valid]
        
        # Discard pixels past 80 m and less than 0.1 m (10 cm)
        depth[depth > 80.0] = 0
        depth[depth < 0.1] = 0
        
        # Save depth map
        save_path = os.path.join(output_dir, sample_name + ".png")
        depth_img = (depth * 256.0).astype(np.uint16)
        cv2.imwrite(save_path, depth_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if (args.dataset == "test"):
            test_images.append(depth_img)
            test_names.append(sample_name)
    
    if (args.dataset == "test"):
        # Create a 5x1 subplot (vertical stack) of the estimated depth maps
        if len(test_images) > 0:
            n = len(test_images)
            fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(8, 3 * n))
            # When n==1, axes is not an array
            if n == 1:
                axes = [axes]
            for i, (img16, name) in enumerate(zip(test_images, test_names)):
                # convert uint16 stored as depth*256 back to meters
                depth = img16.astype(np.float32) / 256.0
                plot = axes[i].imshow(depth, cmap='magma', vmin=0, vmax=80)
                axes[i].set_title(f"Sample: {name}")
                axes[i].axis('off')
                fig.colorbar(plot, ax=axes[i], orientation='vertical', shrink=0.8)
            plt.tight_layout()
            out_path = os.path.join('figures', 'est_depth_stack.png')
            os.makedirs('figures', exist_ok=True)
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

if __name__ == '__main__':
    main()
