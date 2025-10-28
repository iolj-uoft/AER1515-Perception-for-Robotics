import numpy as np
import cv2 as cv
import csv
import os
import argparse
from tools.superglue import SuperGlue

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
    output_file = open("assignment2/Feature_Matching_Correspondence/P3_result.txt", "a")
    output_file.truncate(0)


    ## Main
    # Utilize the SuperGlue framework once for all samples
    SG = SuperGlue()
    if (args.test == 1):
        SG.move_images("assignment2/Feature_Matching_Correspondence/test")
    else:
        SG.move_images("assignment2/Feature_Matching_Correspondence/training")
    
    SG.call_SuperGlue()
    
    images_list = []  # To collect images for vertical stacking
    
    for sample_name in sample_list:
        left_image_path = left_image_dir +'/' + sample_name + '.png'
        right_image_path = right_image_dir +'/' + sample_name + '.png'

        img_left = cv.imread(left_image_path, 0)
        img_right = cv.imread(right_image_path, 0)

        matches = SG.load_data(sample_name) # Load matches for this sample
        
        # Read calibration
        calib_path = calib_dir +'/' + sample_name + '.txt'
        frame_calib = read_frame_calib(calib_path)
        stereo_calib = get_stereo_calibration(frame_calib.p2, frame_calib.p3)

        # Find disparity and depth
        pixel_u_list = [] # x pixel on left image
        pixel_v_list = [] # y pixel on left image
        disparity_list = []
        depth_list = []
        
        for i, match in enumerate(matches):
            u_L = match[0][0]
            v_L = match[0][1]
            u_R = match[1][0]
            v_R = match[1][1]
            
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

        # Ensure at least 100 valid depth pairs before writing
        if len(depth_list) >= 100:
            # Output
            for u, v, disp, depth in zip(pixel_u_list, pixel_v_list, disparity_list, depth_list):
                line = "{} {:.2f} {:.2f} {:.2f} {:.2f}".format(sample_name, u, v, disp, depth)
                output_file.write(line + '\n')
        else:
            print(f"Sample {sample_name}: Only {len(depth_list)} valid depth pairs, skipping output.")
        
        # For plotting, use cv.drawMatches
        if (args.plot == 1):
            # Load the .npz to get keypoints and matches arrays
            npz_path = f"assignment2/Feature_Matching_Correspondence/SuperGlueData/left_{sample_name}_right_{sample_name}_matches.npz"
            if os.path.exists(npz_path):
                npz = np.load(npz_path)
                keypoints0 = npz['keypoints0']
                keypoints1 = npz['keypoints1']
                matches_arr = npz['matches']
                
                # Create KeyPoint objects
                kp_left = [cv.KeyPoint(float(kp[0]), float(kp[1]), 1) for kp in keypoints0]
                kp_right = [cv.KeyPoint(float(kp[0]), float(kp[1]), 1) for kp in keypoints1]
                
                # Create DMatch objects for valid matches
                cv_Dmatches = []
                for i in range(len(matches_arr)):
                    if matches_arr[i] > -1:
                        cv_Dmatches.append(cv.DMatch(i, int(matches_arr[i]), 0, 0))
                
                # Read images
                img_left_bgr = cv.imread(left_image_path)
                img_right_bgr = cv.imread(right_image_path)
                
                # Draw matches
                img_combined = cv.drawMatches(img_left_bgr, kp_left, img_right_bgr, kp_right, cv_Dmatches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                
                # Create canvas with title and white space
                height_title = 50
                height_space = 20
                total_height = height_title + img_combined.shape[0] + height_space
                canvas = np.ones((total_height, img_combined.shape[1], 3), dtype=np.uint8) * 255  # white background
                
                # Add title
                title_text = f"Sample: {sample_name}"
                font = cv.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                font_thickness = 2
                text_size = cv.getTextSize(title_text, font, font_scale, font_thickness)[0]
                text_x = (canvas.shape[1] - text_size[0]) // 2
                cv.putText(canvas, title_text, (text_x, 30), font, font_scale, (0, 0, 0), font_thickness)
                
                # Place the combined image below the title
                canvas[height_title:height_title + img_combined.shape[0], :] = img_combined
                
                # Collect for vertical stacking
                images_list.append(canvas)
            else:
                print(f"NPZ file not found for {sample_name}")

    # Save all plots vertically stacked in one image
    if args.plot == 1 and images_list:
        # Ensure all images have the same width for vstack
        max_width = max(img.shape[1] for img in images_list)
        resized_images = []
        for img in images_list:
            if img.shape[1] != max_width:
                resized_img = cv.resize(img, (max_width, img.shape[0]))
                resized_images.append(resized_img)
            else:
                resized_images.append(img)
        
        combined_image = np.vstack(resized_images)
        out_path = "superglue_matches_all.png"
        cv.imwrite(out_path, combined_image)
        print(f"Saved stacked image {out_path}")
        
        # Optionally display the stacked image
        cv.imshow("All Matches Stacked", combined_image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    output_file.close()