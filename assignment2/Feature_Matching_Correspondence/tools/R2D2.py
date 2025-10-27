import os
from matplotlib import pyplot as plt
import cv2 as cv
import sys
import subprocess
import numpy as np

class R2D2:
    def __init__(self, image_path: str):
        self.image_path = image_path
        
    def inference(self):
        """Inference the R2D2 model with its own extract script
        """
        r2d2_path = os.path.splitext(self.image_path)[0] + '.r2d2' 
        if not os.path.exists(r2d2_path): # only inference if the .npz files DNE
            if self.image_path.endswith('.png'):
                call_r2d2 = [
                    sys.executable,
                    "r2d2/extract.py",
                    "--model", "r2d2/models/faster2d2_WASF_N16.pt",
                    "--images", self.image_path,
                    "--top-k", "1000"
                ]
                subprocess.run(call_r2d2)
                        
    def load_data(self):
        """Load the .r2d2 file associated with the image
        Returns:
            keypoints: (N, 3): x, y, scale (float)
            descriptors: (N, 128): descriptor vectors
            scores: (N, 1): the reliability score from R2D2 paper
        """
        r2d2_path = os.path.splitext(self.image_path)[0] + '.r2d2'
        assert os.path.exists(r2d2_path)
        
        with np.load(r2d2_path) as f:
            keypoints = f['keypoints']
            descriptors = f['descriptors'] 
            scores = f['scores']
            
        return keypoints, descriptors, scores