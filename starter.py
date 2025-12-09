
import numpy as np
import cv2

def load_kitti_image(path):
    """
    Loads a KITTI image from the specified path and converts it from BGR to RGB.
    
    Args:
        path (str): The file path to the image (.png).
        
    Returns:
        np.ndarray: The loaded image as an HxWx3 NumPy array in RGB format.
    """
    # OpenCV loads images in BGR format by default
    bgr_image = cv2.imread(path)
    # Convert from BGR to RGB
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return rgb_image

def load_kitti_lidar_scan(path):
    """
    Loads a KITTI LiDAR scan from a .bin file.
    
    Args:
        path (str): The file path to the LiDAR scan (.bin).
        
    Returns:
        np.ndarray: An Nx4 NumPy array where each row is (x, y, z, reflectance).
    """
    # LiDAR points are stored as a flat array of floats
    scan = np.fromfile(path, dtype=np.float32)
    # Reshape the array to Nx4, where N is the number of points
    points = scan.reshape((-1, 4))
    return points

def load_kitti_calibration(path):
    """
    Loads KITTI calibration data from a .txt file into a dictionary.
    
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
    calib['P2'] = calib['P2'].reshape(3, 4)
    calib['R0_rect'] = calib['R0_rect'].reshape(3, 3)
    calib['Tr_velo_to_cam'] = calib['Tr_velo_to_cam'].reshape(3, 4)
    
    # Pad to Homogenous Form
    pad_row          = np.zeros((1, 3))
    calib['R0_rect'] = np.append(calib['R0_rect'], pad_row, axis=0)
    pad_column       = np.array([[0], [0], [0], [1] ])
    calib['R0_rect'] = np.hstack((calib['R0_rect'], pad_column))
    
    calib['Tr_velo_to_cam'] = np.vstack(calib['Tr_velo_to_cam'], pad_column.T)
    
    return calib

