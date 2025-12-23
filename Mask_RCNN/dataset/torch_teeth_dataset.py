import torch
import numpy as np
from torch.utils.data import Dataset
from Mask_RCNN.dataset.teeth_dataset import TeethDataset
from PIL import Image
from utils.converter import polygons2mask
import cv2

class TorchTeethDataset(Dataset):
    def __init__(self, mrcnn_dataset: TeethDataset, max_size=1333):
        self.mds = mrcnn_dataset       # matterport dataset
        self.max_size = max_size
    def __len__(self):
        return len(self.mds.image_ids)

    def __getitem__(self, idx):
        info = self.mds.image_info[idx]

        # Load image
        image = Image.open(info["path"]).convert("RGB") # np.array(image).shape === (H, W, C) với C = 3 (chanel RGB)
        img_np = np.array(image)
        # Chuyển sang hệ màu LAB để xử lý kênh độ sáng (L) mà không làm thay đổi màu sắc
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        
        # Khởi tạo CLAHE (Cân bằng biểu đồ xám thích nghi)
        # clipLimit: Ngưỡng giới hạn độ tương phản (thường từ 2.0 - 4.0)
        # tileGridSize: Chia ảnh thành các ô nhỏ để xử lý cục bộ (8x8 là chuẩn)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Áp dụng CLAHE lên kênh L (Lightness)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Chuyển ngược lại hệ màu RGB
        img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        image = Image.fromarray(img_np)
        w, h = image.size
        scale = self.max_size/max(w, h)
        new_w, new_h = int(w*scale), int(h*scale)
        image = image.resize((new_w, new_h))
        # Format lại cho PyTorch, từ (H, W, C) thành (C, H, W) và chuẩn hóa về [0.0, 1.0] bằng cách /255.0
        image = torch.tensor(np.array(image)).permute(2, 0, 1) / 255.0

        all_masks = []
        all_boxes = []
        all_labels = []

        for obj in info["objects"]:
            class_id = obj["class_id"]
            
            smoothed_mask_np = polygons2mask(img_shape=(new_h, new_w), polygons=obj["polygons"], scale=scale)
            bbox = obj["bbox"]
            y_min, x_min, y_max, x_max = bbox[0], bbox[1], bbox[2], bbox[3]
            scaled_bbox = [
                x_min * scale,
                y_min * scale,
                x_max * scale,
                y_max * scale
            ]
            
            all_masks.append(smoothed_mask_np)
            all_boxes.append(scaled_bbox)
            all_labels.append(class_id)

        if not all_masks: # If no objects were found
            masks = torch.zeros((0, new_h, new_w), dtype=torch.uint8)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            # Convert lists to tensors
            masks = torch.tensor(np.stack(all_masks)).float()
            boxes = torch.tensor(all_boxes, dtype=torch.float32)
            labels = torch.tensor(all_labels, dtype=torch.int64)
        
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[valid]
            masks = masks[valid]
            labels = labels[valid]

            # Final check for an empty list of objects after validation
            if boxes.size(0) == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                masks = torch.zeros((0, new_h, new_w), dtype=torch.uint8)
                labels = torch.zeros((0,), dtype=torch.int64)
        
        target = {
            "boxes": boxes.float(),
            "labels": labels,
            "masks": masks.float()
            # "image_id": torch.tensor([idx])
        }

        return image, target