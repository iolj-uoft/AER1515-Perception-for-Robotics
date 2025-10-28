import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import csv
import os
import argparse
from tools.R2D2 import R2D2
from tools.matcher import FeatureMatcher

class FrameCalib:
    """Frame Calibration

    Fields:
        p0-p3: (3, 4) Camera P matrices. Contains extrinsic and intrinsic parameters.
        r0_rect: (3, 3) Rectification matrix
        velo_to_cam: (3, 4) Transformation matrix from velodyne to cam coordinate
        Point_Camera = P_cam * R0_rect * Tr_velo_to_cam * Point_Velodyne
        """

    def __init__(self):
        self.p0 = []
        self.p1 = []
        self.p2 = []
        self.p3 = []
        self.r0_rect = []
        self.velo_to_cam = []


def read_frame_calib(calib_file_path):
    """Reads the calibration file for a sample

    Args:
        calib_file_path: calibration file path

    Returns:
        frame_calib: FrameCalib frame calibration
    """

    data_file = open(calib_file_path, 'r')
    data_reader = csv.reader(data_file, delimiter=' ')
    data = []

    for row in data_reader:
        data.append(row)

    data_file.close()

    p_all = []

    for i in range(4):
        p = data[i]
        p = p[1:]
        p = [float(p[i]) for i in range(len(p))]
        p = np.reshape(p, (3, 4))
        p_all.append(p)

    frame_calib = FrameCalib()
    frame_calib.p0 = p_all[0]
    frame_calib.p1 = p_all[1]
    frame_calib.p2 = p_all[2]
    frame_calib.p3 = p_all[3]

    # Read in rectification matrix
    tr_rect = data[4]
    tr_rect = tr_rect[1:]
    tr_rect = [float(tr_rect[i]) for i in range(len(tr_rect))]
    frame_calib.r0_rect = np.reshape(tr_rect, (3, 3))

    # Read in velodyne to cam matrix
    tr_v2c = data[5]
    tr_v2c = tr_v2c[1:]
    tr_v2c = [float(tr_v2c[i]) for i in range(len(tr_v2c))]
    frame_calib.velo_to_cam = np.reshape(tr_v2c, (3, 4))

    return frame_calib


class StereoCalib:
    """Stereo Calibration

    Fields:
        baseline: distance between the two camera centers
        f: focal length
        k: (3, 3) intrinsic calibration matrix
        p: (3, 4) camera projection matrix
        center_u: camera origin u coordinate
        center_v: camera origin v coordinate
        """

    def __init__(self):
        self.baseline = 0.0
        self.f = 0.0
        self.k = []
        self.center_u = 0.0
        self.center_v = 0.0


def krt_from_p(p, fsign=1):
    """Factorize the projection matrix P as P=K*[R;t]
    and enforce the sign of the focal length to be fsign.


    Keyword Arguments:
    ------------------
    p : 3x4 list
        Camera Matrix.

    fsign : int
            Sign of the focal length.


    Returns:
    --------
    k : 3x3 list
        Intrinsic calibration matrix.

    r : 3x3 list
        Extrinsic rotation matrix.

    t : 1x3 list
        Extrinsic translation.
    """
    s = p[0:3, 3]
    q = np.linalg.inv(p[0:3, 0:3])
    u, b = np.linalg.qr(q)
    sgn = np.sign(b[2, 2])
    b = b * sgn
    s = s * sgn

    # If the focal length has wrong sign, change it
    # and change rotation matrix accordingly.
    if fsign * b[0, 0] < 0:
        e = [[-1, 0, 0], [0, 1, 0], [0, 0, 1]]
        b = np.matmul(e, b)
        u = np.matmul(u, e)

    if fsign * b[2, 2] < 0:
        e = [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
        b = np.matmul(e, b)
        u = np.matmul(u, e)

    # If u is not a rotation matrix, fix it by flipping the sign.
    if np.linalg.det(u) < 0:
        u = -u
        s = -s

    r = np.matrix.transpose(u)
    t = np.matmul(b, s)
    k = np.linalg.inv(b)
    k = k / k[2, 2]

    # Sanity checks to ensure factorization is correct
    if np.linalg.det(r) < 0:
        print('Warning: R is not a rotation matrix.')

    if k[2, 2] < 0:
        print('Warning: K has a wrong sign.')

    return k, r, t


def get_stereo_calibration(left_cam_mat, right_cam_mat):
    """Extract parameters required to transform disparity image to 3D point
    cloud.

    Keyword Arguments:
    ------------------
    left_cam_mat : 3x4 list
                   Left Camera Matrix.

    right_cam_mat : 3x4 list
                   Right Camera Matrix.


    Returns:
    --------
    stereo_calibration_info : Instance of StereoCalibrationData class
                              Placeholder for stereo calibration parameters.
    """

    stereo_calib = StereoCalib()
    k_left, r_left, t_left = krt_from_p(left_cam_mat)
    _, _, t_right = krt_from_p(right_cam_mat)

    stereo_calib.baseline = abs(t_left[0] - t_right[0])
    stereo_calib.f = k_left[0, 0]
    stereo_calib.k = k_left
    stereo_calib.center_u = k_left[0, 2]
    stereo_calib.center_v = k_left[1, 2]

    return stereo_calib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, default=0, required=True, help="Select True to perform matching on test dataset.")
    parser.add_argument("--plot", type=int, default=0)
    parser.add_argument("--outlier-rejection", type=str, default="RANSAC", choices=["RANSAC", "epipolar"], required=True, help="Select Matching Algorithm.")
    parser.add_argument("--feature-extractor", type=str, default="R2D2", choices=["R2D2", "ORB"], 
                        help="Select feature extractor: R2D2 or ORB.")
    args = parser.parse_args()
    
    ## Input
    if (args.test == 1):
        left_image_dir = os.path.abspath('assignment2/Feature_Matching_Correspondence/test/left')
        right_image_dir = os.path.abspath('assignment2/Feature_Matching_Correspondence/test/right')
        calib_dir = os.path.abspath('assignment2/Feature_Matching_Correspondence/test/calib')
        sample_list = ['000011', '000012', '000013', '000014','000015']
    else: 
        left_image_dir = os.path.abspath('assignment2/Feature_Matching_Correspondence/training/left')
        right_image_dir = os.path.abspath('assignment2/Feature_Matching_Correspondence/training/right')
        calib_dir = os.path.abspath('assignment2/Feature_Matching_Correspondence/training/calib')
        sample_list = ['000001', '000002', '000003', '000004','000005', '000006', '000007', '000008', '000009', '000010']

    ## Output
    output_file = open("P3_result.txt", "a")
    output_file.truncate(0)


    ## Main
    # Prepare a figure and axes for stacking plots vertically (one row per sample)
    if (args.plot == 1):
        # Create one axis per sample; we'll fill them inside the loop.
        fig, axes = plt.subplots(len(sample_list), 1, figsize=(8, 2 * len(sample_list)))
        # Ensure axes is always indexable as a list
        if len(sample_list) == 1:
            axes = [axes]

    for sample_name in sample_list:
        left_image_path = left_image_dir +'/' + sample_name + '.png'
        right_image_path = right_image_dir +'/' + sample_name + '.png'

        img_left = cv.imread(left_image_path, 0)
        img_right = cv.imread(right_image_path, 0)

        # Initialize a feature detector
        # Inference the R2D2 feature detector and extract keypoints, descriptors, and reliability scores
        if (args.feature_extractor == "R2D2"):
            R2D2_left = R2D2(left_image_path)
            R2D2_right = R2D2(right_image_path)
            
            R2D2_left.inference()
            R2D2_right.inference()
            
            left_image_keypoints, left_image_descriptor, left_image_scores = R2D2_left.load_data()
            right_image_keypoints, right_image_descriptor, right_image_scores = R2D2_right.load_data()
            
            # Perform feature matching
            Matcher = FeatureMatcher(left_image_keypoints, left_image_descriptor, left_image_scores,
                            right_image_keypoints, right_image_descriptor, right_image_scores)
            
            if (args.outlier_rejection == 1):
                matches, descriptor_distances = Matcher.RANSAC_matching(distance_ratio=0.82, distance_threshold=2.0,
                                                                        ransac_reproj_threshold=1, confidence=0.99)
            else:
                matches, descriptor_distances = Matcher.epipolar_matching(distance_ratio=0.82, distance_threshold=2.0)
                
        # Extract keypoints, descriptors using ORB
        if (args.feature_extractor == "ORB"):
            orb = cv.ORB_create(nfeatures=1000)
            
            # Detect and compute for left image
            kp_left, desc_left = orb.detectAndCompute(img_left, None)
            left_image_keypoints = np.array([kp.pt for kp in kp_left], dtype=float)  # (N, 2)
            left_image_descriptor = desc_left 
            left_image_scores = np.array([kp.response for kp in kp_left], dtype=float)  # (N,)
            
            # Detect and compute for right image
            kp_right, desc_right = orb.detectAndCompute(img_right, None)
            right_image_keypoints = np.array([kp.pt for kp in kp_right], dtype=float)  # (N, 2)
            right_image_descriptor = desc_right
            right_image_scores = np.array([kp.response for kp in kp_right], dtype=float)  # (N,)
            
            # Perform feature matching
            Matcher = FeatureMatcher(left_image_keypoints, left_image_descriptor, left_image_scores,
                            right_image_keypoints, right_image_descriptor, right_image_scores)
            
            if (args.outlier_rejection == 1):
                matches, descriptor_distances = Matcher.RANSAC_matching(distance_threshold=45.0, distance_ratio=0.9,
                                                                        ransac_reproj_threshold=1, confidence=0.99)
            else:
                matches, descriptor_distances = Matcher.epipolar_matching(distance_threshold=50.0, distance_ratio=0.95)

        # Handles matches as cv2.DMatch or as list/array of (left_idx, right_idx) pairs.
        def _to_keypoints(kps):
            if len(kps) == 0:
                return []
            if hasattr(kps[0], 'pt'):
                return kps
            return [cv.KeyPoint(float(p[0]), float(p[1]), 1) for p in kps]

        kp_left = _to_keypoints(left_image_keypoints)
        kp_right = _to_keypoints(right_image_keypoints)

        # Read calibration
        calib_path = calib_dir +'/' + sample_name + '.txt'
        frame_calib = read_frame_calib(calib_path)
        stereo_calib = get_stereo_calibration(frame_calib.p2, frame_calib.p3)

        # Find disparity and depth
        pixel_u_list = [] # x pixel on left image
        pixel_v_list = [] # y pixel on left image
        disparity_list = []
        depth_list = []
        cv_Dmatches = []
        
        for i, match in enumerate(matches):
            matched_left_img_idx = match[0]
            matched_right_img_idx = match[1]
            
            u_L = left_image_keypoints[matched_left_img_idx][0]
            v_L = left_image_keypoints[matched_left_img_idx][1]
            u_R = right_image_keypoints[matched_right_img_idx][0]
            v_R = right_image_keypoints[matched_right_img_idx][1]
            
            disparity = u_L - u_R
            if disparity <= 0.1: # avoid divide by 0
                continue
            
            depth = stereo_calib.f * (stereo_calib.baseline) / disparity
            
            # Filter out depths over 80 meters
            if depth > 80 or depth < 0:
                continue
            
            pixel_u_list.append(u_L)
            pixel_v_list.append(v_L)
            disparity_list.append(disparity)
            depth_list.append(depth)    

        # Output
        assert len(depth_list) >= 100 # ensure at least 100 keypoint pairs per sample
        
        for u, v, disp, depth in zip(pixel_u_list, pixel_v_list, disparity_list, depth_list):
            line = "{} {:.2f} {:.2f} {:.2f} {:.2f}".format(sample_name, u, v, disp, depth)
            output_file.write(line + '\n')
           
        
        for i, match in enumerate(matches):
            try:
                lidx = int(match[0])  # Left image keypoint index
                ridx = int(match[1])  # Right image keypoint index
            except Exception:
                continue
            dist = float(descriptor_distances[i]) if (descriptor_distances is not None and i < len(descriptor_distances)) else 0.0
            cv_Dmatches.append(cv.DMatch(_queryIdx=lidx, _trainIdx=ridx, _imgIdx=0, _distance=dist))
        
        # Draw matches
        if (args.plot == 1):
            # Use RGB images for the plot
            img_left_color = cv.imread(left_image_path)
            img_right_color = cv.imread(right_image_path)
            img_left_color = cv.cvtColor(img_left_color, cv.COLOR_BGR2RGB)
            img_right_color = cv.cvtColor(img_right_color, cv.COLOR_BGR2RGB)

            # Draw matches on RGB images
            img = cv.drawMatches(img_left_color, kp_left, img_right_color, kp_right, cv_Dmatches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            # Place this image into the pre-created axis so we control spacing precisely
            idx = sample_list.index(sample_name)
            ax = axes[idx]
            ax.imshow(img, aspect='auto')
            ax.axis('off')
            # Small title with minimal padding to reduce vertical gaps
            ax.set_title(f"Sample: {sample_name}", pad=2, fontsize=9)
    
    if (args.plot == 1):
        plt.subplots_adjust(hspace=0.02, top=0.99, bottom=0.01)
        plt.tight_layout()
        out_path = "matches.png"
        fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0)
        print(f"Saved {out_path}")
        plt.show()

    output_file.close()

