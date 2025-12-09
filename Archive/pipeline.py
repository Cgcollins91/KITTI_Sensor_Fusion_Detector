import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
from starter import  load_kitti_image, load_kitti_lidar_scan
from detector import load_kitti_labels


def get_labels(label_path):
    """
    Reads and loads KITTI object labels from a .txt file (e.g., label_2/{idx}.txt).
    This function acts as the source for 2D object regions and prints the detected cars and their 2D bounding boxes.
    """

    # Load the labels for the specified frame
    detected_cars = load_kitti_labels(label_path)
    
    print(f"Found {len(detected_cars)} cars in '{label_path}'.")
    
    # Print the 2D bounding box for each car
    for i, car in enumerate(detected_cars):
        box = car['bbox_2d']
        print(f"  Car #{i+1}: Bbox [x1, y1, x2, y2] = {box}")
        

def get_file_path(training_path, file_index, type):
    """
    Returns valid path to particular calibration(calib), camera(image_2), lidar(velodyne), 
    or bounding box labels (label_2) given file_index and type
    """
    
    file_types = {'calib'   :'.txt', 
                  'image_2' :'.png', 
                  'velodyne':'.bin',
                  'label_2' :'.txt'}
    
    if type not in file_types.keys():
        print("Error invalid file type requested")
    
    file_path = training_path + type + '/' + f"{file_index:06d}" + file_types[type]
    
    return file_path


def load_kitti_calibration(path):
    """
    Loads KITTI calibration data from .txt file into dictionary.
    Edited Zeid Kotbally provided function to convert matrices to homogenous.
    
    Args:
        path (str): The file path to the calibration file (.txt).
        
    Returns:
        dict: A dictionary containing the calibration matrices.
    """

    calib = {}
    with open(path, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.split(':', 1)
                # Convert the string of numbers into a NumPy array
                calib[key] = np.array([float(x) for x in value.strip().split()])

    # Reshape matrices to their correct dimensions
    calib['P2']             = calib['P2'].reshape(3, 4)
    calib['R0_rect']        = calib['R0_rect'].reshape(3, 3)
    calib['Tr_velo_to_cam'] = calib['Tr_velo_to_cam'].reshape(3, 4)
    
    # Pad R0_rect to Homogenous Form:
    pad_row          = np.zeros((1, 3))
    calib['R0_rect'] = np.append(calib['R0_rect'], pad_row, axis=0)
    pad_column       = np.array([[0], [0], [0], [1] ])
    calib['R0_rect'] = np.hstack((calib['R0_rect'], pad_column))
    
    # Pad Tr_velo_to_cam to Homogenous Form:
    calib['Tr_velo_to_cam'] = np.vstack((calib['Tr_velo_to_cam'], pad_column.T))
    
    return calib


def print_kitti_shapes(img, lidar, calib):
    """
    Prints shapes of Image, LiDAR, and Calibration matrices for Part A (T3, Deliverable).
    
    Inputs:
        img:    HxWx3 numpy array of RGB image
        lidar:  Nx4 numpy array of lidar points (x,y,z,reflectance)
        calib:  Dictionary with calibration matrices
        
    """
    
    print("\n--- Part A: Console Log with Shapes ---")
    print(f"1. Image shape (H x W x 3): {img.shape}")
    print(f"2. LiDAR points shape (N x 4): {lidar.shape}")
    
    print("\n3. Calibration Matrices:")
    print(f"  P2 shape: {calib['P2'].shape}")
    print(f"  R0_rect shape: {calib['R0_rect'].shape}")
    print(f"  Tr_velo_to_cam shape: {calib['Tr_velo_to_cam'].shape}")
    print("--------------------------------------\n")


def visualize_rgb(img, lidar, frame_idx):
    """
    Generates two visualization figures for Part A (T4, Deliverable).
    
    1. Displays the RGB image using matplotlib.
    2. Renders the raw point cloud in Open3D.
    Inputs:
        img:       HxWx3 numpy array of RGB image
        lidar:     Nx4 numpy array of lidar points (x,y,z,reflectance)
        frame_idx: Integer index of the current frame (for titles)
        
    """

    # 1. RGB image
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.title(f"Part A: RGB Image (Frame {frame_idx})")
    plt.axis('off')
    plt.show()

    # 2. Render the raw point cloud in Open3D
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(lidar[:, :3]) # Use only x, y, z columns (ignore reflectance)
        pcd.paint_uniform_color([0.5, 0.5, 0.5])  

        print("Opening Open3D viewer for raw point cloud....")
        o3d.visualization.draw_geometries([pcd], 
                                          window_name=f"Part A: Raw Point Cloud (Frame {frame_idx})",
                                          width=960, height=540)
    except Exception as e:
        print(f"Could not open Open3D viewer (required dependency): {e}")


def project_lidar_to_image(lidar, calib, w, h):
    """ 
    Project lidar points to camera plane, Get Z>0 mask and image bound masks, but do not apply.
    
    Inputs:
        lidar:       Nx4 numpy array of lidar points (x,y,z,reflectance)
        calib:       Dictionary with calibration matrices
        w:           Image width in pixels
        h:           Image height in pixels
    
    Returns:
        uv:          2xN numpy array of pixel coordinates in camera plane
        Z_cam:       1xN numpy array of depth values in camera frame
        bound_mask:  Boolean mask for points within image bounds
        Z_mask:      Boolean mask for points with Z>0 in camera frame
    """
    
    # Convert LiDAR to Homogenous Nx4 Matrice:
    ones_col  = np.ones((len(lidar), 1))
    X_h_velo  = np.hstack((lidar[:,:3], ones_col)).T       # Extract x,y,z and convert to homogenous by adding column of ones, then transpose to 4xN

    # Part B, Task 2
    X_h_cam  = np.dot(calib['Tr_velo_to_cam'], X_h_velo)   # Apply Rigid Transform From LiDAR Frame to Camera Frame
    X_h_rect = np.dot(calib['R0_rect'], X_h_cam)           # Apply Rectification (Rotation) From LiDAR Frame to Camera Frame
    Z_cam    = X_h_rect[2, :]                              # Get LiDar Depth (Z) values in Camera Frame
    
    Z_mask   = Z_cam > 0                                   # Create mask for points with depth (Z) > 0

    Y = np.dot(calib['P2'], X_h_rect)                      # Apply Camera Projection Matrix mappipng rectified camera frame 3-d points to pixels
    u = Y[0, :] / Y[2, :]                                  # X Coordinate of projected LiDAR points in camera frame in pixels
    v = Y[1, :] / Y[2, :]                                  # Y Coordinate of projected LiDAR points in camera frame in pixels
    
    bound_mask = (0 <= u) & (u <= w) & (0 <= v) & (v <= h) # Create Mask for points within image bounds

    uv = np.array((u, v))                                  # Combine u, v into 2xN array of pixel coordinates in camera plane
    
    return uv, Z_cam, bound_mask, Z_mask


def visualize_depth_projection(uv, Z_cam, img, bound_mask, Z_mask, filter=True):
    """
    Plots image overlaid with LiDAR points projected to camera frame, color-coded by depth.
    
    Inputs:
        uv:          2xN numpy array of lidar points projected to cameraplane in pixels
        Z_cam:       1xN numpy array of depth values in camera frame
        img:         HxWx3 numpy array of RGB image
        bound_mask:  Boolean mask for points within image bounds
        Z_mask:      Boolean mask for points with Z>0 in camera frame
        filter:      Boolean flag to filter points outside image bounds and with Z<=0
    """

    if filter:
        uv    = uv[:, bound_mask & Z_mask]
        Z_cam = Z_cam[bound_mask & Z_mask]

    fig, ax1 = plt.subplots(1, figsize=(12, 6))
    
    ax1.imshow(img)      # Plot camera RGB
    scatter = ax1.scatter(
        uv[0, :],        # X (pixels)
        uv[1, :],        # Y (pixels)
        c=Z_cam,         # Depth (for color)
        cmap='jet',      # Color Map
        s=.01,           # Point Size
        alpha=0.5        # Point Transparency
    )
    
    cbar = fig.colorbar(scatter, ax=ax1)   # Add color bar legend for depth
    cbar.set_label('Depth (m)') 

    plt.show()
    

def get_bounding_box_lidar_points(lidar, uv, labels, Z_cam, Z_mask, bound_mask):
    """
    Get Lidar points within file's labeled bounding boxes by applying frustum culling.
    Applies Z>0 and image bound masks.
    
    Inputs:
        lidar:       Nx4 numpy array of lidar points (x,y,z,reflectance)
        uv:          2xN numpy array of lidar points projected to camera plane in pixels
        labels:      List of dictionaries with KITTI label data including 2D bounding boxes
        Z_cam:       1xN numpy array of depth values in camera frame
        Z_mask:      Boolean mask for points with Z>0 in camera frame
        bound_mask:  Boolean mask for points within image bounds
        
    Returns:
        lidar_clusters: List of numpy arrays of lidar points within KITTI labels bounding boxes after applying Z>0 and image bound masks
        Z_clusters:     List of numpy arrays of Z (depth) values for the LiDAR points in each cluster after applying Z>0 and image bound masks

    """

    Z_clusters     = []
    lidar_clusters = []

    for label in labels:
        # Get Bounding Box Points
        bbox = label['bbox_2d']
        u_min, v_min = bbox[0], bbox[1],
        u_max, v_max = bbox[2], bbox[3]
        
        # Create mask of lidar points within bounding box (Apply Frustum Culling)
        u_mask   = (u_min <= uv[0, :]) & (uv[0, :] <= u_max)
        v_mask   = (v_min <= uv[1, :]) & (uv[1, :] <= v_max)
        uv_mask  = u_mask & v_mask

        # Combine Z>0 mask, image bound mask, and bounding box mask
        Z_and_box_mask = uv_mask & Z_mask & bound_mask

        lidar_filter = lidar[Z_and_box_mask, :]
        Z_filter     = Z_cam[Z_and_box_mask]

        # Additional Depth Filtering: Filter for points within 3 meters of median depth
        Z_median          = np.median(Z_filter)
        depth_gate_filter = ( (Z_median - 3) <= Z_filter) & (Z_filter <= (Z_median + 3))

        Z_filter     = Z_filter[depth_gate_filter]
        lidar_filter = lidar_filter[depth_gate_filter, :]

        Z_clusters.append(Z_filter)
        lidar_clusters.append(lidar_filter)


    return lidar_clusters, Z_clusters


def plot_lidar_3d(lidar_clusters):
    """
    Plot Lidar points in 3-D using open3D 
    
    Inputs:
        lidar_clusters: List of numpy arrays of lidar points within KITTI labels bounding boxes
    """

    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name='LiDAR Point Cloud', width=960, height=540, left=5000, top=0) 
    
    for lidar_cluster in lidar_clusters:
        point_cloud        = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(lidar_cluster[:, :3])
        
        vis1.add_geometry(point_cloud)

    while True:
        vis1.update_geometry(point_cloud)
        if not vis1.poll_events():
            break
        vis1.update_renderer()
        
    vis1.destroy_window()


def plot_hist(Z_clusters, lidar_clusters, color='red'):
    """
    Plot Histogram of Lidar X, Y, Z and Depth values.
    
    Inputs:
        Z_clusters:     List of numpy arrays of Z (depth) values for LiDAR points in each cluster
        lidar_clusters: List of numpy arrays of lidar points within KITTI labels bounding boxes
        color:          Color for histogram bars (default 'red')
    
    """
    
    cluster_num = 0
    for Z_cluster, lidar_cluster in zip(Z_clusters, lidar_clusters):
        
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10)) 

        # Plot Histogram of LiDAR X values 
        axes[0,0].hist(lidar_cluster[:,0], bins=50, color=color)
        axes[0,0].set_title(f"Cluster: {cluster_num} LiDAR Point X (m)")
        axes[0,0].set_xlabel('X (meters)')
        axes[0,0].set_ylabel('Frequency')

        # Plot Histogram of LiDAR Y Values
        axes[1,0].hist(lidar_cluster[:,1], bins=50, color=color)
        axes[1,0].set_title(f"Cluster: {cluster_num} LiDAR Point Y (m)")
        axes[1,0].set_xlabel('Y (meters)')
        axes[1,0].set_ylabel('Frequency')
        
        # Plot Histogram of LiDAR Z values
        axes[0,1].hist(lidar_cluster[:,2], bins=50, color=color)
        axes[0,1].set_title(f"Cluster: {cluster_num} LiDAR Point Z (m)")
        axes[0,1].set_xlabel('Z (meters)')
        axes[0,1].set_ylabel('Frequency')

        # Plot Histogram of LiDAR Depth Values
        axes[1,1].hist(Z_cluster, bins=50, color=color)
        axes[1,1].set_title(f"Cluster: {cluster_num} LiDAR Depth")
        axes[1,1].set_xlabel('Depth (meters)')
        axes[1,1].set_ylabel('Frequency')
        
        cluster_num += 1
        
        plt.tight_layout()
        plt.show()


def visualize_2d_detections(img, labels):
    """
    Plots image with 2D bounding boxes overlaid.
    
    Inputs:
        img:    HxWx3 numpy array of RGB image
        labels: List of dictionaries with KITTI label data including 2D bounding boxes
    """

    fig, ax1 = plt.subplots(1, figsize=(12, 6))

    # Plot Camera Image and Bounding Boxes
    ax1.imshow(img)         # Plot camera RGB
    for label in labels:
        bbox = label['bbox_2d']

        rect = patches.Rectangle(
        (bbox[0], bbox[1]), # Bottom Left Corner
        bbox[2]-bbox[0],    # Width
        bbox[3]-bbox[1],    # Height
        linewidth=2,        # Bounding box line width
        edgecolor='b',      # Bounding Box Line Color
        facecolor='none'    # Bounding Box internal color
        )
        ax1.add_patch(rect)
    ax1.set_title("2D Detection Boxes")
    plt.show()
    
    
def get_aabb_box(cluster):
    """
    Compute Axis-Aligned Bounding Box (AABB) for given LiDAR cluster.
    
    Inputs:
        cluster: Numpy array of lidar points within KITTI labels bounding box
        
    Returns:
        box:     Dictionary with bounding box parameters and open3d box object
    """

    pcd        = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cluster[:, :3])

    min_values = pcd.get_min_bound()
    max_values = pcd.get_max_bound()
    dims = max_values - min_values
    center = min_values + dims/2
    aabb_box = o3d.geometry.AxisAlignedBoundingBox(min_values, max_values)
    box = {
        "type"    : "AABB",
        "center_m": center,
        "dims_m"  : {"l": dims[0], "w": dims[1], "h": dims[2]},
        "yaw_rad" : 0.0,
        "box_obj" : aabb_box
    }
    
    return box


def estimate_bounding_boxes(lidar_clusters, obb=False):
    """
    Compute 3D bounding boxes (AABB or OBB) for each LiDAR cluster.
    For OBB, Principal Component Analysis (PCA) is used to determine orientation.
    If PCA fails, defaults to AABB.
    
    Inputs:
        lidar_clusters: List of numpy arrays of lidar points within KITTI labels bounding boxes
        obb:            Boolean flag to compute Oriented Bounding Boxes (True) or Axis-Aligned Bounding Boxes (False)
        
    Returns:
        boxes:          List of dictionaries with bounding box parameters and open3d box object
    """

    boxes = []
    for cluster in lidar_clusters:
        if obb:
            try:
                cluster_3d = cluster[:, :3] # Isolate X, Y, Z component (ignore reflectance)
                
                # Get Mean and Covariance of Cluster:
                centroid   = np.mean(cluster_3d, axis=0)
                cov_matrix = np.cov(cluster_3d, rowvar=False)

                # Get principle components -- Eigenvectors / Eigenvalues
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)    
                
                # Sort Eigenvectors by magnitude of eigenvalues to ensure primary axis is axis with largest variance
                sort_indices    = np.argsort(eigenvalues)[::-1]
                rotation_matrix = eigenvectors[:, sort_indices]

                if np.linalg.det(rotation_matrix) < 0: # Ensure Right Handed Coordinate System, if det <0, rotation matrix is reflection
                    rotation_matrix[:, -1] *= -1       # Flip Direction of last column if left-handed, flipping single axis of left-handed converts back to right-handed

                primary_axis = rotation_matrix[:, 0]   # After sorting by eigenvalue our primary axis is first column

                # Project Cluster points onto PCA axes
                projected_points = (cluster_3d - centroid) @ rotation_matrix

                # Get bounding box min / max after projection along each axes
                min_bound = np.min(projected_points, axis=0)
                max_bound = np.max(projected_points, axis=0)
                dims      = max_bound - min_bound
                
                # Create open3d OBB object:
                obb = o3d.geometry.OrientedBoundingBox(centroid, rotation_matrix, dims)
                
                box = {                       # Create JSON output parmeters for each cluster
                    "type"    : "OBB",
                    "center_m": centroid,
                    "dims_m"  : {"l": dims[0], "w": dims[1], "h": dims[2]},
                    "yaw_rad" : float(np.arctan2(primary_axis[1], primary_axis[0])),
                    "box_obj" : obb,
                    "n_points": len(cluster[:])
                }
                
            except: # if PCA fails, default to AABB
                print("Warning: Could not compute OBB for cluster, defaulting to AABB")
                box = get_aabb_box(cluster)
                
        else:
            box = get_aabb_box(cluster)

        boxes.append(box)
        
    return boxes


def plot_lidar_3d_with_boxes(lidar_clusters, boxes, lidar):
    """
    Plot Lidar clusters in 3-D using open3D with bounding boxes.
    
    Inputs:
        lidar_clusters: List of numpy arrays of lidar points within KITTI labels bounding boxes
        boxes:          List of dictionaries with bounding box parameters and open3d box object
        lidar:          Unfiltered lidar points for background context 
    """

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='3D Boxes', width=960, height=540, left=100, top=100)
    cmap           = plt.get_cmap("jet") 
    cluster_colors = cmap(np.linspace(0, 1, len(lidar_clusters)))[:, :3]
    
    # Plot Background Lidar Points in Gray
    pc_gray        = o3d.geometry.PointCloud()
    pc_gray.points = o3d.utility.Vector3dVector(lidar[:, :3])
    pc_gray.paint_uniform_color([0.5, 0.5, 0.5])

    # Add LiDAR clusters within bounding boxes
    for i, cluster in enumerate(lidar_clusters):
        pc        = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(cluster[:, :3])
        pc.paint_uniform_color(cluster_colors[i])
        vis.add_geometry(pc)

    # Add bounding boxes
    for i, box in enumerate(boxes):
        bbox       = box["box_obj"]
        bbox.color = cluster_colors[i]
        try:
            pc_gray = pc_gray.crop(bbox, invert=True)
        except:
            pass
        vis.add_geometry(bbox)

    vis.add_geometry(pc_gray)

    while True:
        # set default view
        # values found by manually zooming frame 100
        view_controller = vis.get_view_control()

        view_controller.set_zoom(0.041333333333333319)
        view_controller.set_front([ -0.999828848674019, 0.012677640915529876, -0.01347407807616945 ])
        view_controller.set_up([ -0.013376751191763157, 0.0077307828706860348, 0.99988064163867074 ])
        view_controller.set_lookat([ 11.165053455936269, -2.0651555632113086, 1.4834780701741086 ])

        vis.update_geometry(None)
        if not vis.poll_events():
            break
        vis.update_renderer()

    vis.destroy_window()


def project_boxes_to_image(boxes, calib, img):
    """
    Project 3D bounding boxes to camera frame and plot.
    
    Inputs:
        boxes: List of dictionaries with bounding box parameters and open3d box object
        calib: Dictionary with calibration matrices
        img:   HxWx3 numpy array of RGB image
        
    """

    cmap       = plt.get_cmap("jet") 
    box_colors = cmap(np.linspace(0, 1, len(boxes)))[:, :3]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img)

    for box_i, box in enumerate(boxes):
        corners_3d = np.asarray(box["box_obj"].get_box_points())

        ones       = np.ones((corners_3d.shape[0], 1))
        corners_h  = np.hstack((corners_3d, ones))

        # Project without filtering
        X_h_cam  = np.dot(calib['Tr_velo_to_cam'], corners_h.T)
        X_h_rect = np.dot(calib['R0_rect'], X_h_cam)
        Y        = np.dot(calib['P2'], X_h_rect)
        depth    = Y[2,:]
        
        u = Y[0, :] / Y[2, :]
        v = Y[1, :] / Y[2, :]

        # Create edges aligned to open3d box point ordering
        edges = [
            (0, 1), (1, 7), (7, 2), (2, 0),  # Bottom face
            (3, 6), (6, 4), (4, 5), (5, 3),  # Top face
            (0, 3), (1, 6), (7, 4), (2, 5)   # Connecting sides
        ]

        for (i, j) in edges:
            if depth[i] > 0 and depth[j] > 0:
                ax.plot([u[i], u[j]], [v[i], v[j]], color=box_colors[box_i], linewidth=1.5)

    plt.title("Projected 3D Boxes on Image")
    plt.show()


def remove_lidar_outliers(Z_clusters, lidar_clusters):
    """
    Remove outlier points in LiDAR cluster based on Z values (depth).
    Points more than 2 standard deviations away from mean Z value are removed and considered outliers.
    
    Inputs:
        Z_cluster:      Numpy array of Z (depth) values for the LiDAR points in cluster
        lidar_cluster:  Numpy array of LiDAR points within KITTI labels bounding box
    Returns:
        Z_filter:       Numpy array of filtered Z values (outliers removed)
        lidar_filter:   Numpy array of filtered LiDAR points (outliers removed)
    """
    cluster_size = [len(cluster) for cluster in Z_clusters]
    print("Num Points Before Additional Filtering")
    print(cluster_size)

    # Filter lidar outliers based on 2 standard deviations from mean Z
    Z_filter, lidar_filter = [], []

    for Z_cluster, lidar_cluster in zip(Z_clusters, lidar_clusters):
        Z_cluster_avg = np.sum(Z_cluster)/len(Z_cluster)
        Z_cluster_std = np.std(Z_cluster)
        delta         =  2*Z_cluster_std

        Z_avg_mask    = ((Z_cluster_avg - delta) <= Z_cluster) & (Z_cluster <= (Z_cluster_avg + delta))

        lidar_cluster = lidar_cluster[Z_avg_mask, :]
        Z_cluster     = Z_cluster[Z_avg_mask]
        
        Z_filter.append(Z_cluster)
        lidar_filter.append(lidar_cluster)

    cluster_size_filter = [len(cluster) for cluster in Z_filter]
    print("Num Points After Additional Filtering")
    print(cluster_size_filter)
    
    return lidar_filter, Z_filter


# --- MAIN EXECUTION ---
if __name__ == '__main__':

    # Configuration (Set this path to your KITTI training directory)
    working_folder = os.getcwd()
    training_path  = working_folder + '/training/'

    file_index  = int(input("Enter File Index (0-7480): "))
    print("Click Exit on each plot generated to continue script execution")

    # ----------------------------------------------------
    # Part A: Setup & Data Loading
    # ----------------------------------------------------

    # 6.1.2 Get file paths
    img_file   = get_file_path(training_path, file_index, 'image_2')
    calib_file = get_file_path(training_path, file_index, 'calib')
    velo_file  = get_file_path(training_path, file_index, 'velodyne')
    label_file = get_file_path(training_path, file_index, 'label_2')

    # 6.1.3 Load data
    try:
        img    = load_kitti_image(img_file)
        lidar  = load_kitti_lidar_scan(velo_file)
        calib  = load_kitti_calibration(calib_file)
        labels = load_kitti_labels(label_file)
    except FileNotFoundError as e:
        print(f"Error loading data. Check the `training_path` and file index: {e}")
        exit()

    # Test Labels
    get_labels(label_file)

    # 6.1.3 Print shapes
    print_kitti_shapes(img, lidar, calib)

    # 6.1.4 Initial Visualization 
    visualize_rgb(img, lidar, file_index)

    # ----------------------------------------------------
    # Part B: Sensor Calibration and Projection
    # ----------------------------------------------------
    print("\n--- Part B: Sensor Calibration and Projection ---")

    # T0: Parse calibration (handled by load_kitti_calibration)

    # T1-1: Get camera image width and height
    h, w = len(img[:, 0, 0]), len(img[0, :, 0])

    # T1-2, T1-3, T2, T3, T4: Run the projection function (transform, rectify, project, mask generation)
    uv, Z_cam, bound_mask, Z_mask = project_lidar_to_image(lidar, calib, w, h)

    # T5: Validation (LiDAR points projected onto camera Frame)
    print("Visualizing LiDAR points projected onto camera Frame....")
    visualize_depth_projection(uv, Z_cam, img, bound_mask, Z_mask, filter=True)
    print("--------------------------------------\n")

    # ----------------------------------------------------
    # Part C: 2D Detection and 3D Data Association
    # ----------------------------------------------------
    print("\n--- Part C: 2D Detection and 3D Data Association ---")

    # T0, T1, T2, T3: Frustum Culling 
    lidar_clusters, Z_clusters = get_bounding_box_lidar_points(lidar, uv, labels, Z_cam, Z_mask, bound_mask)
    print(f"Isolated {len(lidar_clusters)} 3D clusters (objects detected).")

    # Visualization: Cluster Histograms
    if lidar_clusters:
        plot_hist(Z_clusters, lidar_clusters, color='green')

    # T4: Visualize 2D detection boxes and 3D Association (Isolated Point Clouds)
    print("Plotting camera image with 2D bounding boxes overlaid....")
    visualize_2d_detections(img, labels)

    # Plot lidar points within camera bounding boxes in 3d
    # Lidar points are filtered for depth > 0, within camera image bounds and within bounding box bounds from label file
    if lidar_clusters:
        plot_lidar_3d(lidar_clusters)
    print("--------------------------------------\n")

    # ----------------------------------------------------
    # Part D: 3D Bounding Box Estimation & Visualization
    # ----------------------------------------------------
    print("\n--- Part D: 3D Bounding Box Estimation & Visualization ---")

    # T1: Input and Pre-processing 
    lidar_clusters, Z_clusters = get_bounding_box_lidar_points(lidar, uv, labels, Z_cam, Z_mask, bound_mask)
    lidar_filter, Z_filter     = remove_lidar_outliers(Z_clusters, lidar_clusters) # Remove outliers based on 2 standard deviations from mean Z
    

    # T2: Estimate 3D Box Parameters (AABB or OBB)
    boxes = estimate_bounding_boxes(lidar_filter, obb=False)  # use obb=True if required

    # T3: Finalize and Report Box Parameters
    print("Estimated", len(boxes), "3D boxes")

    # T4: Visualization - 2D Image View and 3D Scene View
    project_boxes_to_image(boxes, calib, img)
    plot_lidar_3d_with_boxes(lidar_filter, boxes, lidar)

    exit()

