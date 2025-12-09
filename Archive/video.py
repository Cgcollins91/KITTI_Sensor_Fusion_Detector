import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import os
from starter  import load_kitti_image, load_kitti_lidar_scan
from detector import load_kitti_labels
from Kitti_Sensor_Fusion_Detector.pipeline import get_labels, get_file_path, load_kitti_calibration, remove_lidar_outliers
from Kitti_Sensor_Fusion_Detector.pipeline import get_bounding_box_lidar_points, project_lidar_to_image, estimate_bounding_boxes

def plot_lidar_3d_with_boxes_video(lidar_clusters, boxes, lidar):
    """
    Plot Lidar points in 3-D using open3D with bounding boxes without displaying the window (for video generation).
    
    Inputs:
        lidar_clusters: List of numpy arrays of lidar points within KITTI labels bounding boxes
        boxes:          List of dictionaries with bounding box parameters and open3d box object
        lidar:          Unfiltered lidar points for background context 
    """

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='3D Boxes', width=1920, height=1080, visible=False)

    cmap           = plt.get_cmap("jet")
    cluster_colors = cmap(np.linspace(0, 1, len(lidar_clusters)))[:, :3]
    
    # Plot Background Lidar Points in Gray
    pc_gray        = o3d.geometry.PointCloud()
    pc_gray.points = o3d.utility.Vector3dVector(lidar[:, :3])
    pc_gray.paint_uniform_color([0.5, 0.5, 0.5])
    vis.add_geometry(pc_gray)

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


    # Set camera view, determined empirically
    view_controller = vis.get_view_control()
    view_controller.set_zoom(0.0413)
    view_controller.set_front([-0.999, 0.012, -0.013])
    view_controller.set_up([-0.013, 0.007, 0.999])
    view_controller.set_lookat([11.16, -2.06, 1.48])

    # Capture the image
    vis.poll_events()                                             # GUI Backend housekeeping
    vis.update_renderer()                                         # Update frame rendering
    lidar_image = vis.capture_screen_float_buffer(do_render=True) # Capture each frame pixel as float [0,1]
    vis.destroy_window()                                          # Close window to free up memory

    # Open3d returns float image in range [0,1], convert to uint8 [0,255] for OpenCV (RGB value)
    return (np.asarray(lidar_image) * 255).astype(np.uint8)


def project_boxes_to_image_video(boxes, calib, img, ax):
    """
    Project 3D bounding boxes to camera image plane and plot without displaying figure (for video generation).
    
    Inputs:
        boxes: List of dictionaries with bounding box parameters and open3d box object
        calib: Dictionary with calibration matrices
        img:   Camera image as HxWx3 numpy array
        
    """
    ax.imshow(img)
    cmap       = plt.get_cmap("jet") 
    box_colors = cmap(np.linspace(0, 1, len(boxes)))[:, :3]

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

    ax.axis('off') # Hide axes ticks
    
    return ax

# --- MAIN EXECUTION ---

if __name__ == '__main__':

    # Configuration (Set this path to your KITTI training directory)
    frame              = int(input("Enter Starting File Index (0-7380)"))             # Get Starting Frame index from user
    video_name         = f"output_video_{frame}_to_{frame+100}.avi"                   # Initial video name (updated later)
    working_folder     = os.getcwd()                                                  # Current working directory           
    training_path      = working_folder + '/training/'                                # Path to KITTI training data
    fourcc             = cv2.VideoWriter.fourcc(*"MJPG")                              # Object to write video
    video_writer       = cv2.VideoWriter(video_name, fourcc, 1, (1920*2, 1080), True) # 1 FPS, 2K resolution (2x1080p side-by-side)
    count_valid_frames = 0                                                            # Count of valid frames processed                                                      
    start_frame        = frame                                                        # Save starting frame for video name
    
    while count_valid_frames < 100:
        # ----------------------------------------------------
        # Part A: Setup & Data Loading
        # ----------------------------------------------------

        # 6.1.2 Get file paths
        img_file   = get_file_path(training_path, frame, 'image_2')
        calib_file = get_file_path(training_path, frame, 'calib')
        velo_file  = get_file_path(training_path, frame, 'velodyne')
        label_file = get_file_path(training_path, frame, 'label_2')
        
        # 6.1.3 Load data
        # ----------------------------------------------------
        # Part B: Sensor Calibration and Projection
        # ----------------------------------------------------
        
        try: # T0: Parse calibration (handled by load_kitti_calibration)
            img    = load_kitti_image(img_file)
            lidar  = load_kitti_lidar_scan(velo_file)
            calib  = load_kitti_calibration(calib_file)
            labels = load_kitti_labels(label_file)
            
        except FileNotFoundError as e:
            print(f"Error loading data. Check the `training_path` and file index: {e}")
            exit()

        if len(labels) == 0: # No labels for this frame, skip it
            print(f"No labels (No Cars Detected) found for index  {frame}, skipping frame.")
            frame += 1
            continue
        
        get_labels(label_file)                      # Print labels for frame
        h, w = len(img[:, 0, 0]), len(img[0, :, 0]) # T1-1: Get camera image width and height

        # T1-2, T1-3, T2, T3, T4: Run the projection function (transform, rectify, project, mask generation)
        uv, Z_cam, bound_mask, Z_mask = project_lidar_to_image(lidar, calib, w, h)

        # ----------------------------------------------------
        # Part C: 2D Detection and 3D Data Association
        # ----------------------------------------------------
        # T0, T1, T2, T3: Frustum Culling and Depth Gating
        lidar_clusters, Z_clusters = get_bounding_box_lidar_points(lidar, uv, labels, Z_cam, Z_mask, bound_mask) # Get lidar points within 2D bounding boxes
        lidar_filter, Z_filter     = remove_lidar_outliers(Z_clusters, lidar_clusters) # Filter lidar outliers based on 2 standard deviations from mean Z
        
        # ----------------------------------------------------
        # Part D: 3D Bounding Box Estimation & Visualization
        # ----------------------- -----------------------------
        # T2: Estimate 3D Box Parameters (AABB or OBB)
        boxes = estimate_bounding_boxes(lidar_filter, obb=False) # obb set to True returns OBB, False returns AABB
        print("Estimated", len(boxes), "3D boxes")               # T3: Finalize and Report Box Parameters

        # T4: Visualization - 2D Image View and 3D Scene View
        lidar_frame = plot_lidar_3d_with_boxes_video(lidar_filter, boxes, lidar)
         
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(38.4, 10.8))  # Create Single Matplotlib figure with two subplots for each Frame
        fig.suptitle(f'Frame: {frame}', fontsize=16)                # Add overall title to figure
        
        ax1 = project_boxes_to_image_video(boxes, calib, img, ax1)  # Plot 2D projection on first subplot
        ax2.imshow(lidar_frame)                                     # Plot 3D Lidar with bounding boxes on second subplot
        ax2.axis('off')                                             # Hide axes ticks       
        plt.tight_layout()                                          # Adjust layout to prevent overlap
        fig.canvas.draw()                                           # Render figure to canvas                                
       
        combined_frame = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)          # Convert Matplotlib figure to BGR for OpenCV
        combined_frame = combined_frame.reshape(fig.canvas.get_width_height()[::-1] + (4,)) # Reshape to HxWx4
        combined_frame = combined_frame[:, :, 1:]                                           # Get only RGB channels (excluding alpha)
        combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)                    # Convert from RGB (Matplotlib) to BGR (OpenCV)

        video_writer.write(combined_frame) # Write the combined frame to video
        plt.close(fig)                     # Close figure to free up memory
        
        count_valid_frames += 1            # Increment count of valid frames processed
        frame              += 1            # Move to next frame index
  
    video_writer.release()                 # Finalize and save video file after processing 100 valid frames
    
    updated_video_name = f"output_video_{start_frame}_to_{frame-1}.avi" # Update video name to reflect actual frames processed (account for skips)
    os.rename(working_folder + '/' + video_name,  working_folder + '/' + updated_video_name)


    exit()