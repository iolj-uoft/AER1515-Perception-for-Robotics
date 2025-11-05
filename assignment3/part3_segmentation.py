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
    depth_dir = 'data/test/gt_depth'
    label_dir = 'data/test/gt_labels'
    output_dir = 'data/test/est_segmentation'
    sample_list = ['000011', '000012', '000013', '000014', '000015']
    ################

    for sample_name in sample_list:
    	# Read depth map
        depth_img_path = os.path.join(depth_dir, sample_name + ".png")
        depth_image = cv2.imread(depth_img_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
        depth_map = depth_image / 256.0

        # Discard depths less than 10cm from the camera

        # Read 2d bbox

        # For each bbox
            # Estimate the average depth of the objects

            # Find the pixels within a certain distance from the centroid

        # Save the segmentation mask
        save_path = os.path.join(output_dir, sample_name + ".png")
        cv2.imwrite(save_path, seg_mask)

if __name__ == '__main__':
    main()
