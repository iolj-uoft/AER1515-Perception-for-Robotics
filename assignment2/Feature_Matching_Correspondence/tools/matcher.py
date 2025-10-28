import numpy as np
from typing import Optional
import cv2 as cv

class FeatureMatcher:
    def __init__(
        self, 
        left_keypoints: np.ndarray, 
        left_descriptors: np.ndarray,
        left_image_scores: np.ndarray,
        right_keypoints: np.ndarray, 
        right_descriptors: np.ndarray,
        right_image_scores: np.ndarray):
        """
        Initialize FeatureMatcher Class

        Args:
            left_keypoints (N, 2 or 3): N keypoints with (x, y, scale (only with R2D2))
            left_descriptors (N, 32 or 128): descriptor vector
            left_image_score (N, 1): score for each keypoint
            right_keypoints (N, 2 or 3): N keypoints with (x, y, scale (only with R2D2))
            right_descriptors (N, 32 or 128): descriptor vector
            right_image_score (N, 1): score for each keypoint
        """
        # Detect if descriptors are binary (uint8) for Hamming distance
        self.is_binary = left_descriptors.dtype == np.uint8
        self.left_descriptors = np.asarray(left_descriptors, dtype=np.uint8 if self.is_binary else float)
        self.right_descriptors = np.asarray(right_descriptors, dtype=np.uint8 if self.is_binary else float)
        
         # helper function to accept (N,2) or (N,3) and pad scale if missing
        def _ensure_kps(kps):
            kps = np.asarray(kps, dtype=float)
            if kps.ndim != 2 or kps.shape[1] not in (2, 3):
                raise ValueError("keypoints must be (N,2) or (N,3)")
            if kps.shape[1] == 2:
                kps = np.hstack([kps, np.ones((kps.shape[0], 1), dtype=float)])  # add scale=1.0
            return kps

        self.left_keypoints = _ensure_kps(left_keypoints)
        self.right_keypoints = _ensure_kps(right_keypoints)

        # helper to normalize scores: None -> ones, scalar -> broadcast, else must match N
        def _ensure_scores(scores, n):
            if scores is None:
                return np.ones((n,), dtype=float)
            s = np.asarray(scores, dtype=float).ravel()
            if s.size == 1:
                return np.full((n,), s.item(), dtype=float)
            if s.size != n:
                raise ValueError("scores length must match keypoints length")
            return s

        self.left_image_scores = _ensure_scores(left_image_scores, self.left_keypoints.shape[0])
        self.right_image_scores = _ensure_scores(right_image_scores, self.right_keypoints.shape[0])
        
        
    
    def epipolar_matching(self, distance_threshold: float = 5.0, distance_ratio: float = 0.9, epsilon_y: float = 2.0):
        """Perform feature matching using brute force with distance ratio test and epipolar constraint

        Args:
            distance_threshold (float): Defaults to 5.0.
            distance_ratio (float): Defaults to 0.9.
            epsilon_y (float): Defaults to 2.0.

        Returns:
            matches (np.ndarrays): keypoint matches (kpL1_idx, kpR1_idx), shape: (# of kp matches, 2)
            distances (np.ndarrays): L2 norm of each pair (d0, d1, ...)
        """
        # Calculate distance for every pair of descriptors, D.shape: (N, N)
        if self.is_binary:
            D = np.zeros((self.left_descriptors.shape[0], self.right_descriptors.shape[0]), dtype=float)
            for i in range(self.left_descriptors.shape[0]):
                xor = self.left_descriptors[i] ^ self.right_descriptors
                D[i] = np.sum(np.unpackbits(xor, axis=1), axis=1)
        else:
            D = np.linalg.norm(
                self.left_descriptors[:, None, :] - self.right_descriptors[None, :, :], axis=2
            )
        
        # Use Lowe's distance ratio to find best pairs:
        # Calculate the two best pairs of right descriptor for each left descriptor
        sorted_idx = np.argsort(D, axis=1) # sort the norm in each row and return its row index. e.g., L1 -> R2
        closest_idx = sorted_idx[:, 0] # (N, 1)
        second_closest_idx = sorted_idx[:, 1] # (N, 1)
        
        d0 = D[np.arange(D.shape[0]), closest_idx] # (N, 1)
        d1 = D[np.arange(D.shape[0]), second_closest_idx] # (N, 1)
        
        left_keypoints = self.left_keypoints
        right_keypoints = self.right_keypoints[closest_idx] # rearrange the right kps
        
        # Perform the ratio test and create a mask
        ratio_mask = (d0 / d1 < distance_ratio) & (d0 <= distance_threshold)
        
        # Perform the epipolar constraint
        epipolar_mask = np.abs(left_keypoints[:, 1] - right_keypoints[:, 1]) < epsilon_y
        
        # Filter out pairs with negative disparities
        disparity_mask = right_keypoints[:, 0] < left_keypoints[:, 0]
        
        mask = ratio_mask & epipolar_mask & disparity_mask
        matches = np.stack([np.where(mask)[0], closest_idx[mask]], axis=1)
        descriptor_distances = d0[mask]  # d0.shape: (# of matches, )
        
        assert matches.shape[0] == descriptor_distances.shape[0]
        
        return matches, descriptor_distances
    
    def RANSAC_matching(self, distance_threshold: float = 5.0, distance_ratio: float = 0.9,
                        ransac_reproj_threshold: float = 1.0, confidence: float = 0.99):
        """Perform feature matching using brute force with distance ratio test and RANSAC for outlier rejection

        Args:
            distance_threshold (float): Defaults to 5.0.
            distance_ratio (float): Defaults to 0.9.
            ransac_reproj_threshold (float): RANSAC reprojection threshold in pixels. Defaults to 1.0.
            confidence (float): RANSAC confidence level. Defaults to 0.99.

        Returns:
            matches (np.ndarrays): keypoint matches (kpL1_idx, kpR1_idx), shape: (# of kp matches, 2)
            distances (np.ndarrays): L2 norm of each pair (d0, d1, ...)
        """
        # Calculate distance for every pair of descriptors, D.shape: (N, N)
        if self.is_binary:
            D = np.zeros((self.left_descriptors.shape[0], self.right_descriptors.shape[0]), dtype=float)
            for i in range(self.left_descriptors.shape[0]):
                xor = self.left_descriptors[i] ^ self.right_descriptors
                D[i] = np.sum(np.unpackbits(xor, axis=1), axis=1)
        else:
            D = np.linalg.norm(
                self.left_descriptors[:, None, :] - self.right_descriptors[None, :, :], axis=2
            )
        
        # Use Lowe's distance ratio to find best pairs:
        # Calculate the two best pairs of right descriptor for each left descriptor
        sorted_idx = np.argsort(D, axis=1) # sort the norm in each row and return its row index. e.g., L1 -> R2
        closest_idx = sorted_idx[:, 0] # (N, 1)
        second_closest_idx = sorted_idx[:, 1] # (N, 1)
        
        d0 = D[np.arange(D.shape[0]), closest_idx] # (N, 1)
        d1 = D[np.arange(D.shape[0]), second_closest_idx] # (N, 1)
        
        left_keypoints = self.left_keypoints
        right_keypoints = self.right_keypoints[closest_idx] # rearrange the right kps
        
        # Perform the ratio test and create a mask
        ratio_mask = (d0 / d1 < distance_ratio) & (d0 <= distance_threshold)
                
        # Filter out pairs with negative disparities
        disparity_mask = right_keypoints[:, 0] < left_keypoints[:, 0]
        mask = ratio_mask & disparity_mask
        
        tentative_matches = np.stack([np.where(mask)[0], closest_idx[mask]], axis=1)
        tentative_distances = d0[mask]
        
        # Not enough matches to run RANSAC
        if tentative_matches.shape[0] < 8:
            return tentative_matches, tentative_distances
        
        # Build point arrays for RANSAC: use the original keypoint arrays
        left_idx = tentative_matches[:, 0].astype(int)
        right_idx = tentative_matches[:, 1].astype(int)
        pts1 = self.left_keypoints[left_idx, :2].astype(np.float32)
        pts2 = self.right_keypoints[right_idx, :2].astype(np.float32)
        
        # Estimate fundamental matrix and inlier mask via RANSAC
        F, inlier_mask = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC,
                                                ransac_reproj_threshold, confidence)
        if inlier_mask is None:
            # fallback: no inliers found
            return tentative_matches, tentative_distances
        
        inlier_mask = inlier_mask.ravel().astype(bool)
        matches = tentative_matches[inlier_mask]
        descriptor_distances = tentative_distances[inlier_mask]
        
        assert matches.shape[0] == descriptor_distances.shape[0]
        
        return matches, descriptor_distances        
        
        
        
        