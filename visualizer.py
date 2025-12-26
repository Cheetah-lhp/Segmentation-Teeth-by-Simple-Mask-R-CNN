import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from Mask_RCNN.dataset import TeethDataset, TorchTeethDataset
import torch, matplotlib, os
from Mask_RCNN.model.mask_rcnn import maskrcnn_resnet50
from torchvision.models.detection import maskrcnn_resnet50_fpn

class TeethVisualizer:
    """
    A utility class to visualize Ground Truth and Model Predictions 
    by integrating the dataset and the model.
    """
    def __init__(self, dataset: TorchTeethDataset, model=None):
        self.dataset = dataset
        self.model = model
        if (self.model):
            self.model.eval()
        self.class_map = self.dataset.mds.class_info
        self.colors = matplotlib.colormaps['hsv']

    def get_color(self, label_id: int):
        return self.colors(label_id % (len(self.class_map) + 1))

    def _get_processed_data(self, idx: int):
        """Fetches and processes data from dataset and model for a given index."""
        
        image_tensor, target = self.dataset[idx]
        # Determine the model's current device (e.g., 'cuda:0' or 'cpu')
        # This is a robust way to find the device.
        model_device = next(self.model.parameters()).device
        
        # --- FIX: Move image tensor to the model's device ---
        image_tensor = image_tensor.to(model_device)
        _, H, W = image_tensor.shape
        print(f"Input Max: {image_tensor.max().item()}, Min: {image_tensor.min().item()}")
        # The model expects a list of tensors (batch of size 1)
        images = [image_tensor]

        # Image (convert back to H, W, C and 0-255 range for plotting)
        image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        
        # Ground Truth (Target)
        gt_masks = target['masks'].cpu().numpy() if 'masks' in target else np.array([])
        gt_boxes = target['boxes'].cpu().numpy()
        gt_labels = target['labels'].cpu().numpy()
        
        # Prediction
        if (self.model):
            with torch.no_grad():
                outputs = self.model(images)
            pred = outputs[0]
            
            num_preds = pred['boxes'].shape[0]
            if num_preds == 0: # Not detect any object
                print("No objects detected by the model.")
                # Initialize all prediction arrays to empty but correct shapes (on CPU)
                pred_masks = np.zeros((0, H, W), dtype=np.uint8)
                pred_boxes = np.zeros((0, 4), dtype=np.float32)
                pred_labels = np.zeros((0,), dtype=np.int64)
                pred_scores = np.zeros((0,), dtype=np.float32)

            else:
                pred_masks_raw = pred['masks'].cpu().numpy()
                pred_masks = (pred_masks_raw > 0.5).astype(np.uint8)
                pred_boxes = pred['boxes'].cpu().numpy()
                pred_labels = pred['labels'].cpu().numpy()
                pred_scores = pred['scores'].cpu().numpy()
            
            return image_np, {
                "gt_masks": gt_masks, "gt_boxes": gt_boxes, "gt_labels": gt_labels,
                "pred_masks": pred_masks, "pred_boxes": pred_boxes, "pred_labels": pred_labels, "pred_scores": pred_scores
            }
        return image_np, {
            "gt_masks": gt_masks, "gt_boxes": gt_boxes, "gt_labels": gt_labels
        }

    def visualize_masks_and_boxes(
        self, 
        idx: int, 
        source: str = 'gt', 
        tooth_index: int = None, 
        score_threshold: float = 0
    ):
        """
        Displays the image with masks and bounding boxes for a specific tooth 
        or all teeth from the chosen source.

        Parameters:
        - idx (int): The index of the image in the dataset.
        - source (str): 'gt' for Ground Truth, 'pred' for Model Prediction.
        - tooth_index (Optional[int]): The index of a specific tooth to display (0-indexed). 
                                       If None, all teeth are displayed.
        - score_threshold (float): Minimum score for filtering predictions (ignored if source='gt').
        """
        
        image_np, data = self._get_processed_data(idx)
        
        if source == 'gt':
            masks = data['gt_masks']
            boxes = data['gt_boxes']
            labels = data['gt_labels']
            scores = None
            title_suffix = "Ground Truth"
        elif source == 'pred':
            # Filter predictions by score threshold
            valid_preds = data['pred_scores'] >= score_threshold
            masks = data['pred_masks'][valid_preds]
            boxes = data['pred_boxes'][valid_preds]
            labels = data['pred_labels'][valid_preds]
            scores = data['pred_scores'][valid_preds]
            title_suffix = f"Prediction (Score > {score_threshold:.2f})"
        else:
            raise ValueError("Source must be 'gt' or 'pred'.")

        # --- Select specific tooth if requested ---
        if tooth_index is not None:
            if 0 <= tooth_index < len(labels):
                # L = [10, 20, 30, 40] -> L[2:3] = [30]; L[2] = 30
                masks = masks[tooth_index:tooth_index+1]
                boxes = boxes[tooth_index:tooth_index+1]
                labels = labels[tooth_index:tooth_index+1]
                scores = scores[tooth_index:tooth_index+1] if scores is not None else None
                title_suffix = f"{title_suffix} | Tooth Index: {tooth_index}"
            else:
                print(f"Warning: Tooth index {tooth_index} out of range (0 to {len(labels)-1}). Displaying all.")
                tooth_index = None # Revert to displaying all
        
        # Display the result
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        self._plot_item(ax, image_np, masks, boxes, labels, f"Image Index {idx} | {title_suffix}", scores)
        plt.show()


    def _plot_item(self, ax: plt.Axes, img_np: np.ndarray, masks: np.ndarray, 
                   boxes: np.ndarray, labels: np.ndarray, title: str, 
                   scores: np.ndarray = None, alpha: float = 0.5):
        """Internal method for plotting the image, masks, boxes, and labels."""
        
        ax.imshow(img_np)
        ax.set_title(title, fontsize=14)
        ax.axis('off')

        num_objects = boxes.shape[0]

        if num_objects > 0:
            for i in range(num_objects):
                mask = masks[i]
                box = boxes[i]
                label = labels[i].item()
                
                color = self.get_color(label)
                
                # --- A. Mask Overlay ---
                # Creates a colored layer and uses the mask array as alpha/intensity
                colored_mask = np.zeros(img_np.shape, dtype=float)
                for c in range(3):
                    colored_mask[:, :, c] = color[c]
                ax.imshow(colored_mask, alpha=mask * alpha)
                
                # --- B. Bounding Box ---
                x_min, y_min, x_max, y_max = box
                width = x_max - x_min
                height = y_max - y_min
                
                rect = Rectangle(
                    (x_min, y_min), width, height, linewidth=2, 
                    edgecolor=color, facecolor='none', linestyle='-'
                )
                ax.add_patch(rect)
                
                # --- C. Label Text ---
                if label > 0 and (label - 1) < len(self.dataset.mds.class_names):
                    label_name = self.dataset.mds.class_names[label - 1]
                else:
                    label_name = f"Unknown_{label}"
                score_text = f" ({scores[i]:.2f})" if scores is not None else ""
                
                ax.text(
                    x_min, y_min - 5, label_name + score_text, 
                    color='white', fontsize=7,
                    bbox=dict(facecolor=color[:3], alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3')
                )

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ROOT_DIR = os.path.abspath("./")
    DIR = os.path.join(ROOT_DIR, "data/Radiographs")
    ANNOTATION_DIR = os.path.join(ROOT_DIR, "data/Segmentation/teeth_polygon.json")
    md = TeethDataset()
    md.load_teeth(DIR, "train", ANNOTATION_DIR)
    md.prepare()
    dataset = TorchTeethDataset(md, max_size=1333)
    
    num_classes = md.num_classes + 1
    # WEIGHTS_PATH = os.path.join(ROOT_DIR, "data/weights_ETE_train/maskrcnn_epoch40.pth")
    # model = maskrcnn_resnet50(pretrained=False, num_classes=num_classes)
    WEIGHTS_PATH = os.path.join(ROOT_DIR, "data/weights_ETE_train/maskrcnn_epoch58.pth")
    model = maskrcnn_resnet50(pretrained=False, num_classes=num_classes)
     
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
    model.to(device)
    
    visualizer = TeethVisualizer(dataset=dataset, model=model)
    #source = 'gt'  ground truth của dataset
    #source = 'pred'  dự đoán của model
    #idx: index của ảnh trong dataset
    #tooth_index: index của răng muốn hiển thị (bắt đầu từ 0). None để hiển thị tất cả răng
    #score_threshold: ngưỡng điểm số để lọc dự đoán (chỉ áp dụng khi source='pred')
    visualizer.visualize_masks_and_boxes(idx=1, source='pred', tooth_index=None, score_threshold=0.8)

    