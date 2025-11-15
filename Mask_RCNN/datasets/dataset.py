import torch
import os
import json
import numpy as np
from PIL import Image
from .generalized_dataset import GeneralizedDataset

class DentalDataset(GeneralizedDataset):
    def __init__(self, root_dir, train):
        super().__init__(max_workers=4, verbose=False)
        self.root_dir = root_dir
        self.train = True

        self.img_dir  =  os.path.join(root_dir, "Radiographs")
        self.mask_max_dir = os.path.join(root_dir, "Segmentation", "maxillomandibular")
        self.mask_teeth_dir = os.path.join(root_dir, "Segmentation", "teeth_mask")
        self.bbox_json = os.path.join(root_dir, "Segmentation", "teeth_bbox.json")

        # Load JSON va gop tat ca label cho cung mot anh
        with open(self.bbox_json, "r") as f:
            data_list = json.load(f)

        # dict: img_id (string bo ".jpg") -> list of objects
        self.bbox_data = {}
        for entry in data_list:
            ext_id = os.path.splitext(entry["External ID"])[0]  # "797" từ "797.jpg"
            objects = entry.get("Label", {}).get("objects", [])
            if ext_id in self.bbox_data:
                self.bbox_data[ext_id].extend(objects)
            else:
                self.bbox_data[ext_id] = objects

        #mapping title tu cac ki tu A, B, C, .. -> int lien tiep :v
        self.title_to_label = {}
        counter = 1  # 0 danh cho background
        for objects in self.bbox_data.values():
            for obj in objects:
                t = obj["title"]
                if t not in self.title_to_label:
                    self.title_to_label[t] = counter
                    counter += 1

        #danh sach id anh (loai bo phan duoi .jpg)
        self.ids = [
            os.path.splitext(f)[0]
            for f in os.listdir(self.img_dir) 
            if f.lower().endswith('.jpg') and os.path.splitext(f)[0] in self.bbox_data]

    def get_image(self, img_id):
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        return image

    def get_target(self, img_id):
        mask_teeth_path = os.path.join(self.mask_teeth_dir, f"{img_id}.jpg")
        mask_teeth = Image.open(mask_teeth_path).convert("L")
        mask_teeth_np = np.array(mask_teeth, dtype=np.uint8)

        """xu ly them mask maxillomandibular (vung ham`) (optional)
        mask_max_path = os.path.join(self.mask_max_dir, f"{img_id}.jpg")
        mask_max = Image.open(mask_max_path).convert("L")
        """

        # objects = self.bbox_data[int(img_id)]["Label"]["objects"]
        # boxes = []
        # labels = []
        # masks_list = []

        # for obj in objects:
        #     title = int(obj["title"])  # nhan rang
        #     bbox = obj["bounding box"]  # bounding box
        #     boxes.append(bbox)
        #     labels.append(title)

        #     # tao mask nhi phan cho rang nay
        #     mask = (mask_teeth_np == title).astype(np.uint8)
        #     masks_list.append(mask)
        
        objects = self.bbox_data[img_id]  # img_id là string, ví dụ "797"
        boxes = []
        labels = []
        masks_list = []
        #chuan hoa mask_teeth_np sang integer de so sanh voi label
        mask_new = np.zeros_like(mask_teeth_np, dtype=np.uint8)
        for obj in objects:
            title = obj["title"]
            label = self.title_to_label[title]
            # Nếu mask_teeth_np chứa số tương ứng title, dùng như sau
            # mask_new[mask_teeth_np == int(title)] = label
            # Nếu mask_teeth_np là nhị phân (0/1), dùng:
            mask_new[mask_teeth_np > 0] = label
        mask_teeth_np = mask_new

        for obj in objects:
            title = obj["title"]
            label = self.title_to_label[title]
            bbox = obj["bounding box"]
            boxes.append(bbox)
            labels.append(label)
            # tao mask nhi phan cho rang nay
            mask = (mask_teeth_np == label).astype(np.uint8)
            masks_list.append(mask) 

        if len(masks_list) == 0:
            # Truong hop k co object
            masks = torch.zeros((0, mask_teeth_np.shape[0], mask_teeth_np.shape[1]), dtype=torch.uint8)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            masks = torch.as_tensor(np.stack(masks_list, axis=0), dtype=torch.uint8)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": torch.tensor([int(img_id)]),
        }
        return target