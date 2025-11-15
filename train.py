import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
# from torchvision import transforms
from Mask_RCNN.datasets.dataset import DentalDataset
import torch.optim as optim
from torch.utils.data import random_split

def collate_fn(batch):
    batch = [b for b in batch if b[1]["boxes"].numel() > 0]  # b[1] là target
    if len(batch) == 0:
        return (), ()
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

        """chuyen target sang GPU"""
        for t in targets:
            t["boxes"] = t["boxes"].to(device)
            t["labels"] = t["labels"].to(device)
            t["masks"] = t["masks"].to(device)

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

    dataset = DentalDataset(
        root_dir=r"C:\Users\Admin\OneDrive\Dokumen\AIOT Lab\My Weekly Report\Teeth Segmentation with Mask R-CNN\Tuft Dental Database",
        #transform=transform,
        train=True
    )

    """Chia 80/20"""
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = n - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])
    """batch_size: so luong sample (anh) duoc dua vao model trong 1 lan forward+backward"""
    train_loader = DataLoader(train_set, batch_size=2, shuffle=True, collate_fn=collate_fn)
    #val_loader = DataLoader(val_set, batch_size=2, shuffle=False, collate_fn=collate_fn)
    
    """so label + 1 background"""
    num_classes = len(dataset.title_to_label) + 1 
    model = maskrcnn_resnet50_fpn(num_classes=num_classes)  
    """1 class (tooth) + background """
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    for epoch in range(num_epochs):
        loss = train_one_epoch(model, optimizer, train_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

        torch.save(model.state_dict(), f"maskrcnn_epoch{epoch+1}.pth")

if __name__ == "__main__":
    main()
