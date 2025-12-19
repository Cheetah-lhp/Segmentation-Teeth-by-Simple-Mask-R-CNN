import torch
import numpy as np
import os, sys
from torch.utils.data import DataLoader
from torchvision import transforms
from Mask_RCNN.model.mask_rcnn import maskrcnn_resnet50
from Mask_RCNN.dataset import TeethDataset, TorchTeethDataset
import torch.optim as optim
from PIL import Image, ImageDraw
from utils.converter import create_binary_smoothed_mask

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return
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

    for data in data_loader:
        if data is None:
            continue
        images, targets = data
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
        #print("\n multi task loss debug")
        #print(loss_dict)
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
    ROOT_DIR = os.path.abspath("./")
    DIR = os.path.join(ROOT_DIR, "data/Radiographs")
    ANNOTATION_DIR = os.path.join(ROOT_DIR, "data/Segmentation/teeth_polygon_chunk_4.json")

    md = TeethDataset()
    md.load_teeth(DIR, "train", ANNOTATION_DIR)
    md.prepare()

    dataset = TorchTeethDataset(md, max_size=1333)
    
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
    num_classes = md.num_classes + 1
    model = maskrcnn_resnet50(pretrained=False, num_classes=num_classes) 

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    num_epochs = 100 
    for epoch in range(num_epochs):
        loss = train_one_epoch(model, optimizer, train_loader, device)
        scheduler.step() 
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}, LR: {current_lr}")

        SAVE_DIR = os.path.join(ROOT_DIR, "data/weights_ETE_train")
        os.makedirs(SAVE_DIR, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"maskrcnn_epoch{epoch+1}.pth"))
        #free VRAM moi epoch
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()