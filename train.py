import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision import transforms
from Mask_RCNN.datasets.dental_dataset import RadiographDataset
import torch.optim as optim
from torch.utils.data import random_split

def collate_fn(batch):
    return tuple(zip(*batch))
"""
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0.0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]

        new_targets = []
        for t in targets:
            mask = torch.from_numpy(np.array(t["mask_teeth"])).long().to(device)

            teeth_ids = torch.unique(mask)
            teeth_ids = teeth_ids[teeth_ids != 0]

            masks = []
            boxes = []
            labels = []

            for tid in teeth_ids:
                m = (mask == tid).float()
                if m.sum() < 10:
                    continue

                y, x = torch.where(m > 0)
                xmin, xmax = x.min().item(), x.max().item()
                ymin, ymax = y.min().item(), y.max().item()

                if xmax <= xmin or ymax <= ymin:
                    continue

                masks.append(m)
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(1)

            if len(masks) == 0:
                new_targets.append(None)
                continue

            new_targets.append({
                "boxes": torch.tensor(boxes, dtype=torch.float32, device=device),
                "labels": torch.tensor(labels, dtype=torch.int64, device=device),
                "masks": torch.stack(masks).to(device)
            })

        # Lọc batch hợp lệ
        valid_images = []
        valid_targets = []
        for img, tgt in zip(images, new_targets):
            if tgt is not None:
                valid_images.append(img)
                valid_targets.append(tgt)

        if len(valid_images) == 0:
            continue

        loss_dict = model(valid_images, valid_targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(data_loader)
"""

def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0.0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]

        # chuyển target sang GPU
        for t in targets:
            t["boxes"] = t["boxes"].to(device)
            t["labels"] = t["labels"].to(device)
            t["masks"] = t["masks"].to(device)

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(data_loader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    dataset = RadiographDataset(
        root_dir="data/Tuft_Dental_Database",
        use_expert=True,
        transform=transform,
        train=True
    )

    # Chia 80/20
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = n - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=2, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False, collate_fn=collate_fn)

    model = maskrcnn_resnet50_fpn(num_classes=2)  # 1 class (tooth) + background
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 10
    for epoch in range(num_epochs):
        loss = train_one_epoch(model, optimizer, train_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

        torch.save(model.state_dict(), f"maskrcnn_epoch{epoch+1}.pth")

if __name__ == "__main__":
    main()
