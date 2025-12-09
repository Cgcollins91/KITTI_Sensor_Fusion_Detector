import numpy as np

def load_kitti_labels(path):
    """
    Loads KITTI-style object detection labels from a .txt file,
    and filters for 'Car' objects.

    Args:
        path (str): The file path to the label file.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary
                    represents a detected 'Car' object and its 2D bbox.
    """
    objects = []
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            obj_type = parts[0]
            
            # For this assignment, we primarily care about cars.
            if obj_type.lower() == 'car':
                bbox = {
                    'type': obj_type,
                    'bbox_2d': np.array([
                        float(parts[4]), # x1 (left)
                        float(parts[5]), # y1 (top)
                        float(parts[6]), # x2 (right)
                        float(parts[7])  # y2 (bottom)
                    ])
                }
                objects.append(bbox)
    return objects

# --- Example Usage ---
if __name__ == '__main__':
    # NOTE: Replace with the actual path to your KITTI data
    label_path = 'training/label_2/000145.txt'
    
    try:
        # Load the labels for the specified frame
        detected_cars = load_kitti_labels(label_path)
        
        print(f"Found {len(detected_cars)} cars in '{label_path}'.")
        
        # Print the 2D bounding box for each car
        for i, car in enumerate(detected_cars):
            box = car['bbox_2d']
            print(f"  Car #{i+1}: Bbox [x1, y1, x2, y2] = {box}")
            
    except FileNotFoundError:
        print(f"Error: Label file not found at '{label_path}'.")
        print("Please update the path to point to your KITTI dataset.")

# --- Sample Output ---
# Found 2 cars in 'training/label_2/000145.txt'.
#   Car #1: Bbox [x1, y1, x2, y2] = [579.41 187.33 614.14 214.15]
#   Car #2: Bbox [x1, y1, x2, y2] = [595.34 179.27 629.41 204.07]