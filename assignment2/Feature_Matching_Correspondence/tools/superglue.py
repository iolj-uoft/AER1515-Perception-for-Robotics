import subprocess
import os
import sys
import shutil
import numpy as np

class SuperGlue:
    def __init__(self):
        pass 
    
    def move_images(self, mother_dir: str, 
                    output_dir: str = "assignment2/Feature_Matching_Correspondence/SuperGlueData/images",
                    pairs_file: str = "assignment2/Feature_Matching_Correspondence/SuperGlueData/pairs.txt"):
        left_img_dir = os.path.join(mother_dir, "left")
        right_img_dir = os.path.join(mother_dir, "right")
        os.makedirs(output_dir, exist_ok=True)
        
        left_files = []
        right_files = []
        
        # Copy and rename PNGs from left directory
        for filename in os.listdir(left_img_dir):
            if filename.endswith('.png'):
                src = os.path.join(left_img_dir, filename)
                dst_name = f"left_{filename}"
                dst = os.path.join(output_dir, dst_name)
                shutil.copy2(src, dst)
                left_files.append(dst_name)
        
        # Copy and rename PNGs from right directory
        for filename in os.listdir(right_img_dir):
            if filename.endswith('.png'):
                src = os.path.join(right_img_dir, filename)
                dst_name = f"right_{filename}"
                dst = os.path.join(output_dir, dst_name)
                shutil.copy2(src, dst)
                right_files.append(dst_name)
        
        with open(pairs_file, 'w') as f:
            for left, right in zip(sorted(left_files), sorted(right_files)):
                f.write(f"{left} {right}\n")

    def call_SuperGlue(self):
        call_superglue = [
                    sys.executable,
                    "assignment2/SuperGluePretrainedNetwork/match_pairs.py",
                    "--superglue", "outdoor",
                    "--output_dir", "assignment2/Feature_Matching_Correspondence/SuperGlueData",
                    "--viz",
                    "--input_pairs", "assignment2/Feature_Matching_Correspondence/SuperGlueData/pairs.txt",
                    "--input_dir", "assignment2/Feature_Matching_Correspondence/SuperGlueData/images",
                    "--resize", "-1", "-1",
                    "--cache",
                    "--max_keypoints", "2000"
                ]
        subprocess.run(call_superglue)
        
    def load_data(self, sample_name: str):
        """
        Load SuperGlue .npz output for a specific sample.
        
        Args:
            sample_name: e.g., '000001'
        
        Returns:
            matched_pairs: List of tuples [(kp0, kp1, confidence), ...] for the sample
        """
        npz_path = f"assignment2/Feature_Matching_Correspondence/SuperGlueData/left_{sample_name}_right_{sample_name}_matches.npz"
        if not os.path.exists(npz_path):
            print(f"Warning: {npz_path} not found, skipping.")
            return []
        
        npz = np.load(npz_path)
        
        keypoints0 = npz['keypoints0']  # (N, 2)
        keypoints1 = npz['keypoints1']  # (M, 2)
        matches = npz['matches']        # (N,) indices in keypoints1 or -1
        match_confidence = npz['match_confidence']  # (N,)
        
        matched_pairs = []
        for i in range(len(matches)):
            if matches[i] > -1:
                kp0 = keypoints0[i]  # (2,)
                kp1 = keypoints1[matches[i]]  # (2,)
                conf = match_confidence[i]  # scalar
                matched_pairs.append((kp0, kp1, conf))
        
        return matched_pairs

    def delete_npz_files(self, output_dir: str = "assignment2/Feature_Matching_Correspondence/SuperGlueData"):
        """
        Delete all .npz files in the specified output directory.
        
        Args:
            output_dir: Directory containing the .npz files to delete.
        """
        import glob
        npz_files = glob.glob(os.path.join(output_dir, '*.npz'))
        for file in npz_files:
            os.remove(file)
        print(f"Deleted {len(npz_files)} .npz files from {output_dir}")


if __name__ == "__main__":
    SG = SuperGlue()
    SG.move_images("assignment2/Feature_Matching_Correspondence/training")
    SG.call_SuperGlue()
    # call_superglue = [
    #                 sys.executable,
    #                 "SuperGluePretrainedNetwork/match_pairs.py",
    #                 "--superglue", "outdoor",
    #                 "--output_dir", "assignment2/Feature_Matching_Correspondence/SuperGlueData",
    #                 "--viz",
    #                 ""
    #             ]
    # subprocess.run(call_superglue)