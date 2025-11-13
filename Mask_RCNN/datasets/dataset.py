import torch
import os
import json
import numpy as np
from PIL import Image
from generalized_dataset import GeneralizedDataset

class DentalDataset(GeneralizedDataset):
    def __init__(self, root_dir, train):
        super().__init__(max_workers=4, verbose=False)
        self.root_dir = root_dir
        self.train = True

        self.img_dir  = os.path.join(root_dir, "Radiographs")
        self.mask_max_dir = os.path.join(root_dir, "Segmentation", "maxillomandibular")
        self.mask_teeth_dir = os.path.join(root_dir, "Segmentation", "teeth_mask")
        self.bbox_json = os.path.join(root_dir, "Segmentation", "teeth_bbox.json")

        #danh sach id anh (loai bo phan duoi .jpg)
        self.ids = [
            os.path.splitext(f)[0]
            for f in os.listdir(self.img_dir) 
            if f.lower().endswith('.jpg')]
        
        with open(os.path.join(self.root_dir, "Segmentation", "teeth_bbox.json")) as f:
            self.bbox_data = json.load(f)


    def get_image(self, img_id):
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        return image
    


    def get_target(self, img_id):
        mask_teeth_path = os.path.join(self.mask_teeth_dir, f"{img_id}.jpg")
        mask_teeth = Image.open(mask_teeth_path).convert("L")
        mask_teeth_np = np.array(mask_teeth)

        """xu ly them mask maxillomandibular (vung ham`) (optional)
        mask_max_path = os.path.join(self.mask_max_dir, f"{img_id}.jpg")
        mask_max = Image.open(mask_max_path).convert("L")
        """

        obj_ids = np.unique(mask_teeth_np)
        obj_ids = obj_ids[1:] #bo qua background class (0)

        #tao mask nhi phan cho tung doi tuong
        masks = mask_teeth_np == obj_ids[:, None, None]

        #doc bounding boxes tu json
        objects = self.bbox_data[int(img_id)]["Label"]["objects"]
        boxes = [obj["bounding box"] for obj in objects]
        labels = [int(obj["title"]) for obj in objects]
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([int(img_id)]),
        }
        return target