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
        
        
    
    def epipolar_matching(self, epsilon_y: float = 2.5):
        D = np.linalg.norm(
            self.left_descriptors[:, None, :] - self.right_descriptors[None, :, :], axis=2
        )
        print(D.shape)
