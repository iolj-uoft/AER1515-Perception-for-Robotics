import os
import sys

import cv2
import numpy as np
import kitti_dataHandler


def main():

    ################
    # Options
    ################
    # Input dir and output dir
    disp_dir = 'data/train/disparity'
    output_dir = 'data/train/est_depth'
    calib_dir = 'data/train/calib'
    sample_list = ['000011', '000012', '000013', '000014', '000015']
    ################

    for sample_name in (sample_list):
        # Read disparity map
        img_path = os.path.join(disp_dir, sample_name + ".png")
        disp = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
        disp /= 256.0

        # Read calibration info

        # Calculate depth (z = f*B/disp)

        # Discard pixels past 80m

        # Save depth map
        save_path = os.path.join(output_dir, sample_name + ".png")
        depth_img = (depth * 256.0).astype(np.uint16)
        cv2.imwrite(save_path, depth_img, [cv2.IMWRITE_PNG_COMPRESSION, 3])


if __name__ == '__main__':
    main()
