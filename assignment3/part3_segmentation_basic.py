import os
import sys
import argparse
import cv2
import numpy as np
import kitti_dataHandler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["train", "test"], default="train", required=True, type=str)
    parser.add_argument("--depth-threshold", default=5.0, required=True, type=float)
    args = parser.parse_args()
    ################
    # Options
    ################
    # Input dir and output dir
    if (args.dataset == "train"):
        depth_dir = 'data/train/est_depth'
        label_dir = 'data/train/gt_labels'
        output_dir = 'data/train/est_segmentation'
        sample_list = ['000001', '000002', '000003', '000004','000005', '000006', '000007', '000008', '000009', '000010']
    else:
        depth_dir = 'data/test/est_depth'
        label_dir = 'data/test/gt_labels'
        output_dir = 'data/test/est_segmentation'
        sample_list = ['000011', '000012', '000013', '000014', '000015']
    ################

    depth_threshold = args.depth_threshold # unit: m
    
    for sample_name in sample_list:
    	# Read depth map
        # Discard depths less than 10cm from the camera -> already done in part 1
        depth_img_path = os.path.join(depth_dir, sample_name + ".png")
        depth_image = cv2.imread(depth_img_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
        depth_map = depth_image / 256.0
        seg_mask = np.full_like(depth_map, 255, dtype=np.uint8)
        
        # Read 2d bbox
        object_list = kitti_dataHandler.read_labels(label_dir, sample_name)
        
        # For each bbox
        for object in object_list:
            # Estimate the average depth of the objects
            x1, x2, y1, y2 = int(object.x1), int(object.x2), int(object.y1), int(object.y2)
            
            # Skip invalid bboxes
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Extract the depth patch
            h, w = depth_map.shape
            x1 = max(0, min(x1, w-1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h-1))
            y2 = max(0, min(y2, h))
            
            depth_patch = depth_map[y1:y2, x1:x2]
            
            # Compute average depth using only valid depths (>0)
            valid_depths = depth_patch[depth_patch > 0]
            if len(valid_depths) == 0:
                continue  # Skip if no valid depths
            avg_depth = np.mean(valid_depths)
            
            # Find the pixels within a certain distance from the average
            depth_threshold_mask = np.abs(depth_patch - avg_depth) <= depth_threshold
            seg_mask[y1:y2, x1:x2] = np.where(depth_threshold_mask, 0, seg_mask[y1:y2, x1:x2])  # Preserve existing values if overlapping
            
        # Save the segmentation mask
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, sample_name + ".png")
        cv2.imwrite(save_path, seg_mask)

if __name__ == '__main__':
    main()
