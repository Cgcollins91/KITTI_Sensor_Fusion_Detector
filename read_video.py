# %%

import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches

import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor



def plot_detected_cars(frame, pred_box=None, pred_score=None):
    """
    Plot Camera Image and Overlay lidar points, 
    if filter set to False image will be hard to read
    """

    x1, y1, x2, y2 = map(int, pred_box) # Convert box coordinates to integers

    # Draw the rectangle (BGR color: Red = (0, 0, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red box with thickness 2

    # Put score text above the box
    label = f"Car: {pred_score:.2f}"
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - h - 5), (x1 + w, y1), (0, 0, 255), -1) # Red background
    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1) # White text
    
    return frame



def get_model(num_classes):
    # Load Pre-Trained Faster R-CNN Model Trainined on Microsoft Common Objects in Context (MS COCO) Dataset
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Get count of input features for classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace pre-trained head with ours (2 classes: 'Car' and 'background')
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def initiate_eval_model(model_path):
    """
    Load trained model from file.
    """
    num_classes = 2  # 1 class (Car) + background
    model       = get_model(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(torch.device('cuda'))
    model.eval()
    
    return model

def predict_image_car_bbox(model, img, conf_thresh=0.5):
    """
    Run object detection model on image and return predicted bounding boxes and labels for 'Car' class.
    """
    car_boxes  = []
    car_scores = []
    # Convert image to tensor
    
    image_tensor = torch.as_tensor(img.transpose((2, 0, 1)), dtype=torch.float32) / 255.0
    image_tensor = image_tensor.to(torch.device('cuda'))
    
    # Get predictions from model
    with torch.no_grad():
        pred = model([image_tensor])[0]

    # Send Predictions back to CPU and convert to numpy
    pred_boxes   = pred['boxes'].cpu().numpy()
    pred_labels  = pred['labels'].cpu().numpy()
    pred_scores  = pred['scores'].cpu().numpy()
    
    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        if label == 1 and score >= conf_thresh:
            car_boxes.append(box)
            car_scores.append(score)
        

    return car_boxes, car_scores

# %%
# if __name__ == '__main__':

# --- Configuration ---
working_folder    = os.getcwd()  
INPUT_VIDEO_PATH  = "youtube.webm"  # <--- CHANGE THIS
OUTPUT_VIDEO_PATH = "output_detector.mp4"    # <--- CHANGE THIS (Optional)
MODEL_PATH        = "car_detector_model.pth"              # Path to your trained model
CONF_THRESHOLD    = 0.5                               # Confidence threshold for detections
DEVICE            = torch.device('cuda')                      # Device to run inference on
model             = initiate_eval_model(MODEL_PATH)

cap = cv2.VideoCapture(INPUT_VIDEO_PATH)            

if not cap.isOpened():
    print(f"Error: Could not open video file {INPUT_VIDEO_PATH}")
    exit()
    
frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps          = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video properties: {frame_width}x{frame_height} @ {fps} FPS, {total_frames} frames")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frame_width, frame_height))

frame_count = 0
while cap.isOpened():
    ret, frame = cap.read() # Reads one frame. 'ret' is True if successful.

    if not ret:
        print("End of video reached or error reading frame.")
        break
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    predicted_boxes, predicted_scores = predict_image_car_bbox(model, frame, conf_thresh=CONF_THRESHOLD)

    # --- Draw Bounding Boxes ---
    # Draw directly on the original BGR frame loaded by OpenCV
    for box, score in zip(predicted_boxes, predicted_scores):
        frame = plot_detected_cars(frame, pred_box=box, pred_score=score)

    out.write(frame)
    
cap.release()
out.release()
cv2.destroyAllWindows() # Close display window if used
    

    # exit()