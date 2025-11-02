import numpy as np
import argparse
import cv2  # For loading .png depth maps
import os  # For file path manipulation

# Updated function to handle dense ground truth depth maps
def calculate_depth_error(gt_depth_dir: str, predicted_depth_file: str):
    """
    Calculate depth estimation errors (MAE, RMSE, Relative Error) between ground truth and predicted depths.

    Args:
        gt_depth_dir (str): Path to the directory containing ground truth depth files (.png).
        predicted_depth_file (str): Path to the predicted depth file (sparse keypoints).

    Returns:
        dict: A dictionary containing MAE, RMSE, and Relative Error.
    """
    # Load predicted depths (sparse keypoints)
    predicted_data = np.loadtxt(predicted_depth_file)  # Shape: (N, 5) [image_id, u, v, disparity, depth]

    # Group predictions by image_id
    unique_image_ids = np.unique(predicted_data[:, 0])
    all_mae, all_rmse, all_relative_error = [], [], []

    for image_id in unique_image_ids:
        # Filter predictions for the current image_id
        image_predictions = predicted_data[predicted_data[:, 0] == image_id]

        # Load the corresponding ground truth depth map
        gt_depth_file = os.path.join(gt_depth_dir, f"{int(image_id):06d}.png")
        if not os.path.exists(gt_depth_file):
            print(f"Warning: Ground truth file {gt_depth_file} not found. Skipping.")
            continue

        gt_depth_map = cv2.imread(gt_depth_file, cv2.IMREAD_UNCHANGED)  # Load as a grayscale image

        # Align ground truth and predicted depths
        gt_depths = []
        predicted_depths = []

        for _, u, v, _, pred_depth in image_predictions:
            u, v = int(u), int(v)  # Ensure pixel coordinates are integers
            if 0 <= v < gt_depth_map.shape[0] and 0 <= u < gt_depth_map.shape[1]:
                gt_depth_raw = gt_depth_map[v, u]  # Access depth at (u, v)
                if gt_depth_raw > 0:  # Ignore invalid depths (e.g., zero or negative values)
                    # Convert ground truth depth from scaled units to meters
                    gt_depth = gt_depth_raw / 256.0
                    gt_depths.append(gt_depth)
                    predicted_depths.append(pred_depth)
                else:
                    continue  # Skip if ground truth depth is 0

        gt_depths = np.array(gt_depths)
        predicted_depths = np.array(predicted_depths)

        # Compute evaluation metrics for the current image
        if len(gt_depths) > 0:
            mae = np.mean(np.abs(predicted_depths - gt_depths))  # Mean Absolute Error
            rmse = np.sqrt(np.mean((predicted_depths - gt_depths) ** 2))  # Root Mean Squared Error
            relative_error = np.mean(np.abs(predicted_depths - gt_depths) / gt_depths)  # Relative Error

            all_mae.append(mae)
            all_rmse.append(rmse)
            all_relative_error.append(relative_error)

    # Aggregate metrics across all images
    return {
        "MAE": np.mean(all_mae) if all_mae else None,
        "RMSE": np.mean(all_rmse) if all_rmse else None,
        "Relative Error": np.mean(all_relative_error) if all_relative_error else None
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate depth estimation errors.")
    parser.add_argument("--gt-depth-dir", type=str, required=True, help="Path to the directory containing ground truth depth files.")
    parser.add_argument("--predicted-depth-file", type=str, required=True, help="Path to the predicted depth file. (P3_result.txt)")

    args = parser.parse_args()

    # Calculate depth errors
    errors = calculate_depth_error(args.gt_depth_dir, args.predicted_depth_file)

    # Print results
    print("Depth Estimation Errors:")
    print(f"Mean Absolute Error (MAE): {errors['MAE']:.4f}")
    print(f"Root Mean Squared Error (RMSE): {errors['RMSE']:.4f}")
    print(f"Relative Error: {errors['Relative Error']:.4f}")