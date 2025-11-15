from .generalized_dataset import GeneralizedDataset
from PIL import Image
import os
import numpy as np
import torch
import json

class RadiographDataset(GeneralizedDataset):
    def __init__(self, root_dir, use_expert=True, transform=None, train=True):
        super().__init__(max_workers=4)
        self.root_dir = root_dir
        self.use_expert = use_expert
        self.transform = transform
        self.train = train

        #chon expert hoac student
        self.person_type = "Expert" if use_expert else "Student"

        #dir toi cac thu muc con
        self.img_dir = os.path.join(root_dir, "Radiographs")
        self.mask_mm_dir = os.path.join(root_dir, "Segmentation", "maxillomandibular")
        self.mask_teeth_dir = os.path.join(root_dir, "Segmentation", "teeth_mask")
        self.gaze_dir = os.path.join(root_dir, self.person_type, "gaze_map", "gray")
        self.expert_mask_dir = os.path.join(root_dir, self.person_type, "mask")

        self.ids = [os.path.splitext(f)[0] for f in os.listdir(self.img_dir)
                    if f.lower().endswith((".jpg", ".png", ".jpes"))]
        self._aspect_ratios = [1.0 for _ in self.ids] #aspect ratio mac dinh

        # #load metadata
        # json_path = os.path.join(root_dir, self.person_type, f"{self.person_type()}.json")
        # self.metadata = json.load(open(json_path)) if os.path.exists(json_path) else {}

    def get_image(self, img_id):
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        img = Image.open(img_path).convert("RGB")
        return img
    
    def get_target(self, img_id):
        target = {}

        mask_mm_path = os.path.join(self.mask_mm_dir, f"{img_id}.jpg")
        mask_teeth_path = os.path.join(self.mask_teeth_dir, f"{img_id}.jpg")
        expert_mask_path = os.path.join(self.expert_mask_dir, f"{img_id}.jpg")
        gaze_path = os.path.join(self.gaze_dir, f"{img_id}.jpg")

        mask_mm = Image.open(mask_mm_path).convert("L") if os.path.exists(mask_mm_path) else None
        mask_teeth = Image.open(mask_teeth_path).convert("L") if os.path.exists(mask_teeth_path) else None
        expert_mask = Image.open(expert_mask_path).convert("L") if os.path.exists(expert_mask_path) else None
        gaze_map = Image.open(gaze_path).convert("L") if os.path.exists(gaze_path) else None

        #chuyen mask thanh tensor
        mask_teeth_np = np.array(mask_teeth)
        obj_ids = np.unique(mask_teeth_np)
        obj_ids = obj_ids[1:] #bo qua background class (0)

        masks = mask_teeth_np == obj_ids[:, None, None]
        boxes = []

        for i in range(len(obj_ids)):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            ymin = np.min(pos[0])
            xmax = np.max(pos[1])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(obj_ids), ), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        #luu vao target dict
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = torch.tensor([int(img_id)])
        target["mask_mm"] = mask_mm
        target["mask_teeth"] = mask_teeth
        target["expert_mask"] = expert_mask
        target["gaze_map"] = gaze_map

        return target
    
    def __getitem__(self, idx):
        img_id = self.ids[idx]
        image = self.get_image(img_id)
        target = self.get_target(img_id)

        if self.transform:
            image = self.transform(image)

            # Resize mask theo image size
            masks = target["masks"].unsqueeze(1).float()
            masks = torch.nn.functional.interpolate(
                masks, size=(image.shape[1], image.shape[2]), mode="nearest"
            ).squeeze(1).long()
            target["masks"] = masks

            # Cap nhat lai boxes theo mask da resize
            boxes = []
            valid_masks = []
            for m in masks:
                pos = torch.where(m > 0)
                if len(pos[0]) == 0:   # mask rỗng → bỏ qua
                    continue

                xmin, xmax = pos[1].min(), pos[1].max()
                ymin, ymax = pos[0].min(), pos[0].max()

                # loại mask có box không hợp lệ (pixel quá ít)
                if xmax <= xmin or ymax <= ymin:
                    continue

                boxes.append([xmin, ymin, xmax, ymax])
                valid_masks.append(m)

            # Nếu không còn mask nào hợp lệ → return ảnh đó nhưng không mask → Mask R-CNN sẽ skip
            if len(boxes) == 0:
                return image, {
                    "boxes": torch.zeros((0,4), dtype=torch.float32),
                    "labels": torch.zeros((0,), dtype=torch.int64),
                    "masks": torch.zeros((0, image.shape[1], image.shape[2]), dtype=torch.uint8)
                }

            target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
            target["masks"] = torch.stack(valid_masks)
            target["labels"] = torch.ones((len(valid_masks),), dtype=torch.int64)

        return image, target
    