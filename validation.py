import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from train import TorchTeethDataset, collate_fn
from Mask_RCNN.dataset import teeth_dataset

from visualize import visualize_prediction

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ROOT_DIR = r"C:\Users\Admin\OneDrive\Dokumen\AIOT Lab\My Weekly Report\Teeth Segmentation with Mask R-CNN\data"
    DIR = ROOT_DIR + "/Radiographs"
    ANN = ROOT_DIR + "/Segmentation/teeth_polygon_chunk_4.json"

    md = teeth_dataset.TeethDataset()
    md.load_teeth(DIR, "val", ANN)
    md.prepare()

    dataset = TorchTeethDataset(md, max_size=512)

    model = maskrcnn_resnet50_fpn(num_classes=md.num_classes)
    model.load_state_dict(torch.load("maskrcnn_epoch10.pth", map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    image, target = dataset[0]
    with torch.no_grad():
        pred = model([image.to(device)])[0]
        print(pred.keys())
        print(pred["boxes"].shape)
        print(pred["scores"][:5])
        print(pred["masks"].shape)

    visualize_prediction(image, pred, score_thresh=0.03, topk=100)

if __name__ == "__main__":
    main()