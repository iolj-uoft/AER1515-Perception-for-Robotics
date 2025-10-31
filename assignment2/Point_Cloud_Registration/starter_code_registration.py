import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.spatial import cKDTree
import os

def load_point_cloud(path: str):
    # Load the point cloud data (do NOT change this function!)
    data = pd.read_csv(path, header=None)
    point_cloud = data.to_numpy()
    return point_cloud


def nearest_search(pcd_source: np.ndarray, pcd_target: np.ndarray):
    """Perform NN search to find correspondence of two point clouds using the kdtree data structure

    Args:
        pcd_source (np.ndarray): source point cloud, (N, 3)
        pcd_target (np.ndarray): target point cloud, (N, 3)

    Returns:
        pcd_source (np.ndarray): source point cloud
        corr_target (np.ndarray): corresponding target point cloud
        ec_dist_mean (float): mean of error sum for each correspondence
    """
    
    kdtree = cKDTree(pcd_target)
    distances, idxs = kdtree.query(pcd_source, k=1)
    
    corr_target = pcd_target[idxs]
    ec_dist_mean = np.mean(distances)
    
    return pcd_source, corr_target, ec_dist_mean


def estimate_pose(corr_source: np.ndarray, corr_target: np.ndarray):
    """Estimate the pose between two correlated point clouds using SVD

    Args:
        corr_source (np.ndarray): corresponded source point cloud
        corr_target (np.ndarray): corresponded target point cloud

    Returns:
        pose: a 4x4 SE(3) matrix
        translation_{x, y, z}: the t componenet from the pose matrix, used for plotting
    """
    # Compute the 6D pose (4x4 transform matrix)
    # Get the 3D translation (3x1 vector)
    pose = np.identity(4)
    
    source_pc_mean = np.mean(corr_source, axis=0)
    target_pc_mean = np.mean(corr_target, axis=0)
    source_pc_decentralized = corr_source - source_pc_mean
    target_pc_decentralized = corr_target - target_pc_mean
    
    W = source_pc_decentralized.T @ target_pc_decentralized
    
    U, _, Vt = np.linalg.svd(W)
    R = Vt.T @ U.T
    
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
        
    translation = target_pc_mean - R @ source_pc_mean
    pose[:3, :3] = R
    pose[:3, 3] = translation
    
    translation_x = translation[0]
    translation_y = translation[1]
    translation_z = translation[2]
    
    return pose, translation_x, translation_y, translation_z


def icp(pcd_source: np.ndarray, pcd_target: np.ndarray, name: str = None, fig_dir: str = None):
    """Perform the ICP algorithm and produce plots for analysis

    Args:
        pcd_source (np.ndarray): the source point cloud
        pcd_target (np.ndarray): the target point cloud
        name (str, optional): the name of the object, used for plotting. Defaults to None.
        fig_dir (str, optional): the directory of saved plots. Defaults to None.

    Returns:
        pose: a 4x4 SE(3) Matrix
    """
    # Put all together, implement the ICP algorithm
    # Use your implemented functions "nearest_search" and "estimate_pose"
    # Run 30 iterations
    pose = np.identity(4)
    corr_source = pcd_source.copy()
    max_iter = 30

    # diagnostics to plot
    mean_dist_per_iter = []
    trans_x_per_iter = []
    trans_y_per_iter = []
    trans_z_per_iter = []

    for i in range(max_iter):
        corr_source, corr_target, ec_dist_mean = nearest_search(corr_source, pcd_target)
        pose_increment, translation_x, translation_y, translation_z = estimate_pose(corr_source, corr_target)

        # accumulate pose (T_total = T_inc @ T_total)
        pose = pose_increment @ pose

        # apply incremental transform correctly: x' = R_inc @ x + t_inc
        R_inc = pose_increment[:3, :3]
        t_inc = pose_increment[:3, 3]
        corr_source = (R_inc @ corr_source.T).T + t_inc

        # record diagnostics
        mean_dist_per_iter.append(ec_dist_mean)
        trans_x_per_iter.append(translation_x)
        trans_y_per_iter.append(translation_y)
        trans_z_per_iter.append(translation_z)

        if ec_dist_mean < 1e-4:
            break

    # ensure fig_dir and name exist
    if fig_dir is None:
        fig_dir = "figures"
    os.makedirs(fig_dir, exist_ok=True)
    if name is None:
        name = "icp"

    # Plot mean euclidean distance per iteration
    plt.figure()
    plt.plot(range(1, len(mean_dist_per_iter) + 1), mean_dist_per_iter, marker='None', color='blue')
    plt.xlabel('Number of Iterations')
    plt.ylabel('ICP Loss (Mean Euclidean Distance)')
    plt.title(f'ICP: Mean Euclidean Distance per Iteration — {name}')
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, f"{name}_icp_mean_distance.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # Plot translation components per iteration
    plt.figure()
    iters = range(1, len(trans_x_per_iter) + 1)
    plt.plot(iters, trans_x_per_iter, marker='None', color='red', label='Translation X')
    plt.plot(iters, trans_y_per_iter, marker='None', color='green', label='Translation Y')
    plt.plot(iters, trans_z_per_iter, marker='None', color='blue', label='Translation Z')
    plt.xlabel('Number of Iterations')
    plt.ylabel('3D Translation (mm)')
    plt.title(f'ICP: Translation per Iteration — {name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, f"{name}_icp_translation_components.png"), dpi=300, bbox_inches='tight')
    plt.show()

    return pose




def main():
    FIG_DIR = "figures"
    os.makedirs(FIG_DIR, exist_ok=True)
    # Dataset and ground truth poses
    #########################################################################################
    # Training and test data (3 pairs in total)
    train_file = ['bunny', 'dragon']
    test_file = ['armadillo']

    # Ground truth pose (from training data only, used for validating your implementation)
    GT_poses = []
    gt_pose = [0.8738,-0.1128,-0.4731,24.7571,
            0.1099,0.9934,-0.0339,4.5644,
            0.4738,-0.0224,0.8804,10.8654,
            0.0,0.0,0.0,1.0]
    gt_pose = np.array(gt_pose).reshape([4,4])
    GT_poses.append(gt_pose)
    gt_pose = [0.7095,-0.3180,0.6289,46.3636,
               0.3194,0.9406,0.1153,3.3165,
               -0.6282,0.1191,0.7689,-6.4642,
               0.0,0.0,0.0,1.0]
    gt_pose = np.array(gt_pose).reshape([4,4])
    GT_poses.append(gt_pose)
    #########################################################################################



    # Training (validate your algorithm)
    ##########################################################################################################
    for i, object in enumerate(train_file):
        # Load data
        path_source = './training/' + object + '_source.csv'
        path_target = './training/' + object + '_target.csv'
        pcd_source = load_point_cloud(path_source)
        pcd_target = load_point_cloud(path_target)
        gt_pose_i = GT_poses[i]

        # Visualize the point clouds before the registration
        ax = plt.axes(projection='3d')
        ax.scatter3D(pcd_source[:,0], pcd_source[:,1], pcd_source[:,2], c ='green')
        ax.scatter3D(pcd_target[:,0], pcd_target[:,1], pcd_target[:,2], c ='red')
        plt.legend(["Source Point Cloud" , "Target Point Cloud"])
        ax.set_title('Point Clouds Before Registration')
        plt.savefig(os.path.join(FIG_DIR, f"{object}_before_registration.png"), dpi=300)
        plt.show()



        # Use your implemented ICP algorithm to get the estimated 6D pose (from source to target point cloud)
        pose = icp(pcd_source, pcd_target, name=object, fig_dir=FIG_DIR)

        # Transform the point cloud
        pts = np.vstack([np.transpose(pcd_source), np.ones(len(pcd_source))])
        cloud_registered = np.matmul(pose, pts)
        cloud_registered = np.transpose(cloud_registered[0:3, :])

        # Evaluate the rotation and translation error of your estimated 6D pose with the ground truth pose
        # compute rotation error (degrees) and translation error (L2 norm)
        R_est = pose[:3, :3]
        t_est = pose[:3, 3]

        R_gt = gt_pose_i[:3, :3]
        t_gt = gt_pose_i[:3, 3]

        R_diff = R_est @ R_gt.T
        cos_angle = (np.trace(R_diff) - 1.0) / 2.0
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        rot_err_rad = np.arccos(cos_angle)
        rot_err_deg = np.degrees(rot_err_rad)

        trans_err = np.linalg.norm(t_est - t_gt)

        print(f"{object} -- Rotation error (deg): {rot_err_deg:.4f}, Translation error: {trans_err:.4f}")

        # Visualize the point clouds after the registration
        ax = plt.axes(projection='3d')
        ax.scatter3D(cloud_registered[:,0], cloud_registered[:,1], cloud_registered[:,2], c ='green')
        ax.scatter3D(pcd_target[:,0], pcd_target[:,1], pcd_target[:,2], c ='red')
        plt.legend(["Transformed Source Point Cloud", "Target Point Cloud"])
        ax.set_title('Point Clouds After Registration')
        plt.savefig(os.path.join(FIG_DIR, f"{object}_after_registration.png"), dpi=300)
        plt.show()
    ##########################################################################################################



    # Test
    ####################################################################################
    for i in range(1):
        # Load data
        path_source = './test/' + test_file[i] + '_source.csv'
        path_target = './test/' + test_file[i] + '_target.csv'
        pcd_source = load_point_cloud(path_source)
        pcd_target = load_point_cloud(path_target)

        # Visualize the point clouds before the registration
        ax = plt.axes(projection='3d')
        ax.scatter3D(pcd_source[:,0], pcd_source[:,1], pcd_source[:,2], c ='green')
        ax.scatter3D(pcd_target[:,0], pcd_target[:,1], pcd_target[:,2], c ='red')
        plt.legend(["Source Point Cloud" , "Target Point Cloud"])
        ax.set_title('Point Clouds Before Registration')
        plt.savefig(os.path.join(FIG_DIR, f"{test_file[i]}_before_registration.png"), dpi=300)
        plt.show()

        # Use your implemented ICP algorithm to get the estimated 6D pose (from source to target point cloud)
        pose = icp(pcd_source, pcd_target, name=test_file[i], fig_dir=FIG_DIR)

        # Calculate the estimated 6D pose (4x4 transformation matrix)
        np.set_printoptions(precision=6, suppress=True)
        print(f"{test_file[i]} -- Estimated 6D pose:\n{pose}")
        pose_txt_path = os.path.join(FIG_DIR, f"{test_file[i]}_estimated_pose.txt")
        np.savetxt(pose_txt_path, pose, fmt="%.6f")

        # Visualize the registered point cloud and the target point cloud
        pts = np.vstack([pcd_source.T, np.ones(len(pcd_source))])
        cloud_registered = (pose @ pts)[:3, :].T

        ax = plt.axes(projection='3d')
        ax.scatter3D(cloud_registered[:,0], cloud_registered[:,1], cloud_registered[:,2], c='green')
        ax.scatter3D(pcd_target[:,0], pcd_target[:,1], pcd_target[:,2], c='red')
        plt.legend(["Transformed Source Point Cloud", "Target Point Cloud"])
        ax.set_title(f'Point Clouds After Registration — {test_file[i]}')
        plt.savefig(os.path.join(FIG_DIR, f"{test_file[i]}_after_registration.png"), dpi=300)
        plt.show()
    ####################################################################################
    
if __name__ == "__main__":
    main()