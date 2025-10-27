import numpy as np

class Matcher:
    def __init__(
        self, 
        left_keypoints: np.ndarray, 
        left_descriptors: np.ndarray,
        left_image_scores: np.ndarray,
        right_keypoints: np.ndarray, 
        right_descriptors: np.ndarray,
        right_image_scores: np.ndarray):
        """
        Initialize Matcher Class

        Args:
            left_keypoints (N, 3): N keypoints with (x, y, scale)
            left_descriptors (N, 128): descriptor vector
            left_image_scores (N, 1): reliability scores for each keypoint
            right_keypoints (N, 3): N keypoints with (x, y, scale)
            right_descriptors (N, 128): descriptor vector
            right_image_scores (N, 1): reliability scores for each keypoint
        """
        
        self.left_keypoints = left_keypoints
        self.left_descriptors = left_descriptors
        self.left_image_scores = left_image_scores
        self.right_keypoints = right_keypoints
        self.right_descriptors = right_descriptors
        self.right_image_scores = right_image_scores
        
        
    
    def epipolar_matching(self, distance_threshold: float = 5.0, distance_ratio: float = 0.9, epsilon_y: float = 2):
        """Perform feature matching using brute force with distance ratio test and epipolar constraint

        Args:
            distance_threshold (int, optional): Defaults to 10.
            distance_ratio (float, optional): Defaults to 0.9.
            epsilon_y (float, optional): Defaults to 2.5.

        Returns:
            matches (np.ndarrays): keypoint matches (kp1_idx, kp2_idx) -> (# of kp matches, 2)
            distances (np.ndarrays): L2 norm of each pair (d0, d1, ...)
        """
        # Calculate L2 norm for every pair of descriptors, D.shape: (N, N)
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
        right_keypoints = self.right_keypoints[closest_idx]
        
        # Perform the ratio test and mask out the pair if test failed
        ratio_mask = (d0 / d1 < distance_ratio) & (d0 <= distance_threshold)
        
        # Perform the epipolar constraint
        epipolar_mask = np.abs(left_keypoints[:, 1] - right_keypoints[:, 1]) < epsilon_y
        disparity_mask = right_keypoints[:, 0] < left_keypoints[:, 0]
        
        mask = ratio_mask & epipolar_mask & disparity_mask
        matches = np.stack([np.where(mask)[0], closest_idx[mask]], axis=1)
        distances = d0[mask]  # d0.shape: (# of filtered left desc, )
        
        assert d0.shape == closest_idx.shape
        
        return matches, distances
        
        
        
        
        