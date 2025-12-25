import torch
import numpy as np
from torch.utils.data import Dataset
from Mask_RCNN.dataset.teeth_dataset import TeethDataset
from PIL import Image
import cv2
from utils.converter import polygons2mask
from Mask_RCNN.dataset.augmentor import TeethAugmentor

class TorchTeethDataset(Dataset):
    def __init__(self, mrcnn_dataset: TeethDataset, max_size=1333):
        self.mds = mrcnn_dataset       # matterport dataset
        self.max_size = max_size
        self.rare_augmentor = TeethAugmentor()

    def __len__(self):
        return len(self.mds.image_ids)

    def __getitem__(self, idx):
        info = self.mds.image_info[idx]

        # 1. Load image bằng OpenCV để xử lý NumPy (tương thích Albumentations)
        image_raw = cv2.imread(info["path"])
        image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image_raw.shape[:2]

        # 2. Áp dụng CLAHE 
        lab = cv2.cvtColor(image_raw, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # 3. Chuẩn bị dữ liệu gốc để Augment
        orig_masks = []
        orig_bboxes = []
        orig_labels = []

        for obj in info["objects"]:
            # Tạo mask ở kích thước gốc (scale=1)
            m = polygons2mask(img_shape=(h_orig, w_orig), polygons=obj["polygons"], scale=1.0)
            orig_masks.append(m)
            
            # Bbox format Pascal_VOC cho Albumentations: [x_min, y_min, x_max, y_max]
            y_min, x_min, y_max, x_max = obj["bbox"]
            orig_bboxes.append([x_min, y_min, x_max, y_max])
            orig_labels.append(obj["class_id"])

        # 4. THỰC HIỆN AUGMENTATION
        # (Nếu không có object nào, bỏ qua augment để tránh lỗi)
        if len(orig_bboxes) > 0:
            img_np, aug_masks, aug_bboxes, aug_labels = self.rare_augmentor(image=img_np, masks=orig_masks, bboxes=orig_bboxes, labels=orig_labels)
        else:
            aug_masks, aug_bboxes, aug_labels = orig_masks, orig_bboxes, orig_labels

        # 5. RESIZE (Sau khi đã augment)
        h_aug, w_aug = img_np.shape[:2]
        scale = self.max_size / max(w_aug, h_aug)
        new_w, new_h = int(w_aug * scale), int(h_aug * scale)
        
        # Resize ảnh
        image_resized = cv2.resize(img_np, (new_w, new_h))
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0

        all_masks = []
        all_boxes = []
        all_labels = []

        # Resize masks và bboxes theo tỉ lệ mới
        for m, b, l in zip(aug_masks, aug_bboxes, aug_labels):
            # Resize mask bằng INTER_NEAREST để giữ tính nhị phân
            m_resized = cv2.resize(m, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            scaled_bbox = [
                b[0] * scale, # x_min
                b[1] * scale, # y_min
                b[2] * scale, # x_max
                b[3] * scale  # y_max
            ]
            
            all_masks.append(m_resized)
            all_boxes.append(scaled_bbox)
            all_labels.append(l)

        if not all_masks:
            masks = torch.zeros((0, new_h, new_w), dtype=torch.uint8)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            masks = torch.tensor(np.stack(all_masks)).float()
            boxes = torch.tensor(all_boxes, dtype=torch.float32)
            labels = torch.tensor(all_labels, dtype=torch.int64)
        
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[valid]
            masks = masks[valid]
            labels = labels[valid]

            if boxes.size(0) == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                masks = torch.zeros((0, new_h, new_w), dtype=torch.uint8)
                labels = torch.zeros((0,), dtype=torch.int64)
        
        target = {
            "boxes": boxes.float(),
            "labels": labels,
            "masks": masks.float()
        }

        if len(all_boxes) == 0:
            return None
            
        return image_tensor, target