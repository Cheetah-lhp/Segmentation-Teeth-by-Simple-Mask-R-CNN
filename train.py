import torch
import numpy as np
import os, sys
from torchvision.ops import masks_to_boxes
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models.detection import maskrcnn_resnet50_fpn
# from torchvision import transforms
from Mask_RCNN.dataset import teeth_dataset
import torch.optim as optim
from PIL import Image, ImageDraw

class TorchTeethDataset(Dataset):
    def __init__(self, mrcnn_dataset, max_size=512):
        self.mds = mrcnn_dataset       # matterport dataset
        self.max_size = max_size
    def __len__(self):
        return len(self.mds.image_ids)

    def __getitem__(self, idx):
        info = self.mds.image_info[idx]

        # Load image
        image = Image.open(info["path"]).convert("RGB")
        w, h = image.size
        scale = self.max_size/max(w, h)
        new_w, new_h = int(w*scale), int(h*scale)
        image = image.resize((new_w, new_h))
        image = torch.tensor(np.array(image)).permute(2, 0, 1) / 255.0

        masks = []
        boxes = []
        labels = []

        for obj in info["objects"]:
            class_id = obj["class_id"]

            for poly in obj["polygons"]:
                if len(poly) < 3:
                    continue
                # tao mask tu polygon
                scaled_poly = [(p[0] * scale, p[1] * scale) for p in poly]

                mask = Image.new("L", (new_w, new_h), 0)
                ImageDraw.Draw(mask).polygon(scaled_poly, outline=1, fill=1)
                mask = torch.tensor(np.array(mask), dtype=torch.uint8)

                masks.append(mask)
                labels.append(class_id)

        if len(masks) == 0:
            masks = torch.zeros((0, new_h, new_w), dtype=torch.uint8)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            masks = torch.stack(masks)
            boxes = masks_to_boxes(masks)
            labels = torch.tensor(labels, dtype=torch.int64)
        
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
            "masks": masks,
            "image_id": torch.tensor([idx])
        }

        return image.float(), target

def collate_fn(batch):
    # batch = [b for b in batch if b[1]["boxes"].numel() > 0]  # b[1] là target
    # if len(batch) == 0:
    #     return (), ()
    return tuple(zip(*batch))
"""ham bat buoc co trong cac bai segmentation nhieu vat the:
        dua 1 batch tu dang: 
            [
                (image1, target1),
                (image2, target2),
                (image3, target3)
            ]
        ve 1 tensor duy nhat:
            (
                (image1, image2, image3),      # tuple chua cac anh
                (target1, target2, target3)    # tuple chua cac target
            )
"""

def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0.0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        """dictionary chua cac loss cua model:
        {
            'loss_classifier' (head)
            'loss_box_reg'
            'loss_mask'
            'loss_objectness' (RPN)
            'loss_rpn_box_reg' (RPN)
        }
        """
        multi_task_loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        multi_task_loss.backward()
        optimizer.step()

        total_loss += multi_task_loss.item()

    return total_loss / len(data_loader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # transform = transforms.Compose([
    #     transforms.Resize((512, 512)),
    #     transforms.ToTensor()
    # ])
    ROOT_DIR = os.path.abspath(r"C:\Users\Admin\OneDrive\Dokumen\AIOT Lab\My Weekly Report\Teeth Segmentation with Mask R-CNN\data")
    sys.path.append(ROOT_DIR)
    DIR = os.path.join(ROOT_DIR, "Radiographs")
    ANNOTATION_DIR = os.path.join(ROOT_DIR, "Segmentation/teeth_polygon_chunk_4.json")

    md = teeth_dataset.TeethDataset()
    md.load_teeth(DIR, "train", ANNOTATION_DIR)
    md.prepare()

    dataset = TorchTeethDataset(md, max_size=512)
    
    # """Chia 80/20"""
    # n = len(dataset)
    # n_train = int(0.8 * n)
    # n_val = n - n_train
    # train_set, val_set = random_split(dataset, [n_train, n_val])
    train_set = dataset
    """batch_size: so luong sample (anh) duoc dua vao model trong 1 lan forward+backward"""
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=collate_fn)
    #val_loader = DataLoader(val_set, batch_size=2, shuffle=False, collate_fn=collate_fn)
    
    """so label + 1 background"""
    num_classes = md.num_classes
    model = maskrcnn_resnet50_fpn(num_classes=num_classes) 
    #tat resize trong mask rcnn de giam RAM GPU
    model.transform.min_size = (512,)
    model.transform.max_size = 512
    model.transform.image_mean = [0.0, 0.0, 0.0]
    model.transform.image_std = [1.0, 1.0, 1.0]

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    for epoch in range(num_epochs):
        loss = train_one_epoch(model, optimizer, train_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

        torch.save(model.state_dict(), f"maskrcnn_epoch{epoch+1}.pth")
        #free VRAM moi epoch
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()