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

        image_pil = Image.open(info["path"]).convert("RGB")
        img_np = np.array(image_pil)
        
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        h_ori, w_ori = img_np.shape[:2]
        scale = self.max_size / max(h_ori, w_ori)
        new_w, new_h = int(w_ori * scale), int(h_ori * scale)
        img_resized = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        all_masks = []
        all_boxes = []
        all_labels = []

        for obj in info["objects"]:
            class_id = obj["class_id"]
            # Tạo mask
            smoothed_mask_np = polygons2mask(img_shape=(new_h, new_w), polygons=obj["polygons"], scale=scale)
            
            # Bbox format [x_min, y_min, x_max, y_max]
            bbox = obj["bbox"]
            scaled_bbox = [
                bbox[1] * scale, # x_min
                bbox[0] * scale, # y_min
                bbox[3] * scale, # x_max
                bbox[2] * scale  # y_max
            ]
            
            all_masks.append(smoothed_mask_np)
            all_boxes.append(scaled_bbox)
            all_labels.append(class_id)

        img_aug, masks_aug, boxes_aug, labels_aug = self.rare_augmentor(img_resized, all_masks, all_boxes, all_labels)

        # Chuyển từ (H, W, C) -> (C, H, W) và chia 255.0
        image_tensor = torch.from_numpy(img_aug).permute(2, 0, 1).float() / 255.0

        if not masks_aug: 
            masks = torch.zeros((0, new_h, new_h), dtype=torch.float32)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            masks = torch.as_tensor(np.stack(masks_aug), dtype=torch.float32)
            boxes = torch.as_tensor(boxes_aug, dtype=torch.float32)
            labels = torch.as_tensor(labels_aug, dtype=torch.int64)

            # Loại bỏ các box lỗi sau khi biến dạng (nếu có)
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[valid]
            masks = masks[valid]
            labels = labels[valid]

            if boxes.size(0) == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                masks = torch.zeros((0, new_h, new_w), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks
        }

        return image_tensor, target