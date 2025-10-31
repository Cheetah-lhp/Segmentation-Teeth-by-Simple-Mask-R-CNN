import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision import transforms
from Mask_RCNN.datasets.dental_dataset import RadiographDataset
from torchvision.ops import box_iou
import numpy as np

def evaluate_iou(pred_masks, true_masks):
    pred = pred_masks > 0.5
    true = true_masks > 0.5

    intersection = (pred & true).float().sum((1, 2))
    union = (pred | true).float().sum((1, 2))
    iou = (intersection / (union + 1e-6)).mean().item()
    return iou

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    dataset = RadiographDataset(
        root_dir="data/Tuft Dental Database",
        use_expert=True,
        transform=transform,
        train=False
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = maskrcnn_resnet50_fpn(num_classes=2)
    model.load_state_dict(torch.load("maskrcnn_epoch10.pth", map_location=device))
    model.eval().to(device)

    ious = []
    with torch.no_grad():
        for images, targets in loader:
            image = images[0].to(device)
            output = model([image])[0]

            if len(output["masks"]) == 0:
                continue

            pred_mask = output["masks"][0, 0].cpu()
            true_mask = transforms.ToTensor()(targets[0]["mask_teeth"])

            iou = evaluate_iou(pred_mask, true_mask)
            ious.append(iou)

    print(f"Mean IoU: {np.mean(ious):.4f}")

if __name__ == "__main__":
    main()
