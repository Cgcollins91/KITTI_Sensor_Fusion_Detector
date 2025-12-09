# %%
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from starter import load_kitti_image
from detector import load_kitti_labels
import numpy as np


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


class KittiDataset(Dataset):
    def __init__(self, training_path, file_indices, transforms=None):
        """
        Args:
            training_path (str): Path to the 'training/' folder.
            file_indices (list): List of file indices (e.g., [0, 1, 2, ..., 150, ...]) to include in this dataset.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.training_path = training_path
        self.file_indices  = file_indices
        self.transforms    = transforms

    def __len__(self):
        return len(self.file_indices)

    def __getitem__(self, idx):
        
        file_index = self.file_indices[idx] # Get the file index for this sample
        
        # --- Get File paths and load data ---
        img_file    = get_file_path(self.training_path, file_index, 'image_2')
        label_file  = get_file_path(self.training_path, file_index, 'label_2')
        image       = load_kitti_image(img_file)        # Returns HxWx3 NumPy array
        labels_list = load_kitti_labels(label_file)
        # ------------------------------------

        # --- Convert data to the format the model needs ---
        boxes  = []
        labels = []
        
        for item in labels_list:
            # We are only training on 'Car' for this example
            if item['type'] == 'Car':
                boxes.append(item['bbox_2d'])
                labels.append(1) # 1 = 'Car' (0 is usually 'background')

        # Convert to tensors
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        if len(boxes) == 0: # Create an empty tensor shape [0, 4]
            boxes_tensor = torch.empty((0, 4), dtype=torch.float32)
        else:
            boxes_np     = np.array(boxes)
            boxes_tensor = torch.as_tensor(boxes_np, dtype=torch.float32)
        
        # Create 'target' dictionary
        target = {}
        target["boxes"]    = boxes_tensor
        target["labels"]   = labels
        target["image_id"] = torch.tensor([idx])

        # Convert image from HWC (Numpy) to CHW (PyTorch tensor)
        # and normalize RGB to [0, 1]
        image_tensor = torch.as_tensor(image.transpose((2, 0, 1)), dtype=torch.float32) / 255.0

        # TBD introduce transforms later
        # if self.transforms:
        #     image_tensor, target = self.transforms(image_tensor, target)
            
        return image_tensor, target
    
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT") # Initialize with Resnet50 FPN Model

    # Get count of input features for classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features # Get count of input features in base model (ResNet50)
    
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    return model

# Function to unpack target dictionaries in DataLoader
def collate_fn(batch):
   return tuple(zip(*batch))

# %%
training_path = '/home/cgcollins91/Dropbox/projects/ENPM818Z_On_Road_Automated_Vehicles/EMPM818Z_FALL_2025_RWA_1/training/'
all_indices     = list(range(7481))
train_indices   = all_indices[:6000]
val_indices     = all_indices[6000:]
batch_size      = 18
num_epochs      = 5
num_classes     = 2  # (1 for 'Car' + 1 for 'background')
device          = torch.device('cuda')
model           = get_model(num_classes)
backbone_params = []
head_params     = []
model.to(device)

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    
    if 'backbone' in name:           # Check if parameter is in 'backbone'
        backbone_params.append(param)
    else:                            # All other parameters are in the RPN or ROI heads
        head_params.append(param)

# Create parameter groups for optimizer
param_groups = [
    {'params': backbone_params, 'lr': 0.0005},  # Low LR for the backbone
    {'params': head_params,     'lr': 0.005}    # High LR for the new head
]

# ---------------------   Load Data / Initialize Model   --------------------------------------------
train_dataset = KittiDataset(training_path, train_indices)
val_dataset   = KittiDataset(training_path, val_indices)
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  collate_fn=collate_fn)
val_loader    = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

model       = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT") # Initialize with Resnet50 FPN Model
in_features = model.roi_heads.box_predictor.cls_score.in_features                     # Get count of input features in base model (ResNet50)
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)           # Replace pre-trained head with our 2 classes: 'Car' and 'background')
# ---------------------------------------------------------------------------------------------------

# ---------------------   Training    ---------------------------------------------------------------
optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=0.0005)
model.train() # Set model to train mode

for epoch in range(num_epochs):
    print(f"--- Epoch {epoch+1}/{num_epochs} ---")
    epoch_loss = 0.0
    for images, targets in train_loader:
        images  = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets) #  During Training, model returns dict of losses
        losses    = sum(loss for loss in loss_dict.values())

        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        epoch_loss += losses.item()
    print(f"Epoch {epoch+1} Avg Loss: {epoch_loss/len(train_loader)}")
    
    epoch_save_path = f"car_detector_epoch_{epoch+1}.pth"
    torch.save(model.state_dict(), epoch_save_path)
    print(f"*** Epoch {epoch+1} finished. Checkpoint saved to {epoch_save_path} ***")

print("--- Training Finished ---")
torch.save(model.state_dict(), "car_detector_model.pth")
# -----------------------------------------------------------------------------------------------

# %%
# from thop import profile




# macs, params = profile(model, inputs=(images,targets ))

# print(f"--- Model (ResNet-50 + FPN) ---")
# print(f"Total parameters (weights): {params / 1_000_000:.2f} M")
# print(f"Total MACs (Giga-MACs): {macs / 1_000_000_000:.2f} G")
# %%
