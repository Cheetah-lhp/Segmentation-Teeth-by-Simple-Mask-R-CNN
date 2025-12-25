import torch
import numpy as np
from torch.utils.data import Dataset
from Mask_RCNN.dataset.teeth_dataset import TeethDataset
from PIL import Image
from utils.converter import polygons2mask
import cv2
from Mask_RCNN.dataset.augmentor import RareTeethAugmentor

class TorchTeethDataset(Dataset):
    def __init__(self, mrcnn_dataset: TeethDataset, max_size=1333):
        self.mds = mrcnn_dataset
        self.max_size = max_size
        self.rare_augmentor = RareTeethAugmentor()

    def __len__(self):
        return len(self.mds.image_ids)

    def __getitem__(self, idx):
        info = self.mds.image_info[idx]

        # 1. Load image
        image = Image.open(info["path"]).convert("RGB")
        img_np = np.array(image)
        
        # Áp dụng CLAHE 
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        image = Image.fromarray(img_np)
        
        w, h = image.size
        scale = self.max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h))
        image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1) / 255.0

        all_masks = []
        all_boxes = []
        all_labels = []

        for obj in info["objects"]:
            # QUAN TRỌNG: polygons2mask nhận obj["polygons"] (là một list các polygon)
            # Hàm này sẽ vẽ tất cả các mảnh của 1 chiếc răng vào DUY NHẤT 1 mask.
            # Điều này đảm bảo 1 Object = 1 Mask.
            mask_np = polygons2mask(img_shape=(new_h, new_w), polygons=obj["polygons"], scale=scale)
            
            bbox = obj["bbox"] # [y_min, x_min, y_max, x_max]
            scaled_bbox = [
                bbox[1] * scale, # x_min
                bbox[0] * scale, # y_min
                bbox[3] * scale, # x_max
                bbox[2] * scale  # y_max
            ]
            
            # Append đồng thời để đảm bảo độ dài các list luôn bằng nhau
            all_masks.append(mask_np)
            all_boxes.append(scaled_bbox)
            all_labels.append(obj["class_id"])

        if not all_masks:
            masks = torch.zeros((0, new_h, new_w), dtype=torch.uint8)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            # Chuyển list thành tensor
            masks = torch.tensor(np.stack(all_masks)).float()
            boxes = torch.tensor(all_boxes, dtype=torch.float32)
            labels = torch.tensor(all_labels, dtype=torch.int64)
        
            # Lọc các box hợp lệ (tránh lỗi box có diện tích bằng 0)
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            
            # Bây giờ masks, boxes và labels đều có cùng số lượng phần tử ở chiều index 0
            boxes = boxes[valid]
            masks = masks[valid]
            labels = labels[valid]

            if boxes.size(0) == 0:
                masks = torch.zeros((0, new_h, new_w), dtype=torch.uint8)
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
        
        target = {
            "boxes": boxes.float(),
            "labels": labels,
            "masks": masks.float()
        }

        return image_tensor, target