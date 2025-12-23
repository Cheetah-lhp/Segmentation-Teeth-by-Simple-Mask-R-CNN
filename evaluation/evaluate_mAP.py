import torch
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
PROJECT_ROOT = FILE.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torchvision.ops import box_iou
from Mask_RCNN.model.mask_rcnn import maskrcnn_resnet50
from Mask_RCNN.dataset import teeth_dataset
from Mask_RCNN.dataset.torch_teeth_dataset import TorchTeethDataset
from ETE_train import collate_fn


# --- 1. CÁC HÀM TÍNH TOÁN mAP ---

def calculate_ap_per_class(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    """
    Tính Average Precision (AP) và trả về cả Precision/Recall arrays.
    """
    if len(gt_boxes) == 0:
        # Nếu không có GT, trả về mảng rỗng hoặc giá trị mặc định
        return 0.0, np.array([0., 1.]), np.array([0., 0.]) 
    
    if len(pred_boxes) == 0:
        return 0.0, np.array([0., 1.]), np.array([0., 0.])

    # 1. Sắp xếp dự đoán theo điểm tin cậy giảm dần
    sorted_indices = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[sorted_indices]
    
    # 2. Tính IoU
    ious = box_iou(pred_boxes, gt_boxes)
    
    tp = torch.zeros(len(pred_boxes))
    fp = torch.zeros(len(pred_boxes))
    gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)

    # 3. Xác định TP/FP
    for i in range(len(pred_boxes)):
        iou_max, gt_idx = torch.max(ious[i], dim=0)
        if iou_max >= iou_threshold and not gt_matched[gt_idx]:
            tp[i] = 1
            gt_matched[gt_idx] = True
        else:
            fp[i] = 1

    # 4. Tính Precision và Recall tích lũy
    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)
    
    recalls = tp_cumsum / len(gt_boxes)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    
    # 5. Làm mượt
    precisions = torch.cat((torch.tensor([1.0]), precisions, torch.tensor([0.0])))
    recalls = torch.cat((torch.tensor([0.0]), recalls, torch.tensor([1.0])))
    
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = torch.max(precisions[i], precisions[i + 1])
        
    indices = torch.where(recalls[1:] != recalls[:-1])[0]
    ap = torch.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return ap.item(), precisions.numpy(), recalls.numpy()

def evaluate_map(model, data_loader, device, num_classes, iou_threshold=0.5):
    model.eval()
    
    # Lưu trữ dữ liệu
    class_data = {i: {'pred_boxes': [], 'pred_scores': [], 'gt_boxes': []} for i in range(1, num_classes + 1)}
    
    print("Đang thu thập dữ liệu để tính mAP...")
    with torch.no_grad():
        for batch in tqdm(data_loader): 
            if batch is None: continue 
            images, targets = batch
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            for i, output in enumerate(outputs):
                p_boxes = output["boxes"].cpu()
                p_scores = output["scores"].cpu()
                p_labels = output["labels"].cpu()
                g_boxes = targets[i]["boxes"].cpu()
                g_labels = targets[i]["labels"].cpu()
                
                for cls_id in range(1, num_classes + 1):
                    cls_mask_p = (p_labels == cls_id)
                    class_data[cls_id]['pred_boxes'].append(p_boxes[cls_mask_p])
                    class_data[cls_id]['pred_scores'].append(p_scores[cls_mask_p])
                    cls_mask_g = (g_labels == cls_id)
                    class_data[cls_id]['gt_boxes'].append(g_boxes[cls_mask_g])

    # Tính AP và lưu Precision/Recall cho từng class
    aps = []
    pr_data = {} # Dictionary lưu p, r cho từng class
    
    # print("\n--- KẾT QUẢ AP TỪNG CLASS ---")
    for cls_id in range(1, num_classes + 1):
        p_boxes = torch.cat(class_data[cls_id]['pred_boxes']) if class_data[cls_id]['pred_boxes'] else torch.tensor([])
        p_scores = torch.cat(class_data[cls_id]['pred_scores']) if class_data[cls_id]['pred_scores'] else torch.tensor([])
        g_boxes = torch.cat(class_data[cls_id]['gt_boxes']) if class_data[cls_id]['gt_boxes'] else torch.tensor([])
        
        ap, prec, rec = calculate_ap_per_class(p_boxes, p_scores, g_boxes, iou_threshold)
        
        aps.append(ap)
        pr_data[cls_id] = {"precision": prec, "recall": rec, "ap": ap}
        
    #     print(f"Class {cls_id}: AP@{iou_threshold} = {ap:.4f}")
        
    mAP = np.mean(aps)
    return mAP, aps, pr_data

# --- 2. HÀM VẼ BIỂU ĐỒ AP & PR CURVE ---

def plot_map_results(aps_input, pr_data, class_names, save_dir="evaluation/evaluation_results"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    mAP = np.mean(aps_input)
    
    # --- BIỂU ĐỒ 1: BAR CHART (Lọc để vẽ) ---
    labels = []
    aps = []
    for i, ap in enumerate(aps_input):
        if ap > 0:
            name = class_names[i] 
            labels.append(name)
            aps.append(ap)

    if not aps:
        print("Không có class nào có AP > 0 để vẽ Bar Chart.")
        return
    
    plt.figure(figsize=(15, 6))
    bars = plt.bar(labels, aps, color='skyblue', edgecolor='navy')
    
    # Vẽ đường kẻ đỏ dựa trên mAP thực tế (ví dụ: 0.4632)
    plt.axhline(y=mAP, color='r', linestyle='--', label=f'Overall mAP: {mAP:.4f}')
    
    plt.title('Average Precision (AP) per Class (Only AP > 0)')
    plt.xlabel('Tooth Class')
    plt.ylabel('AP Score')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=90, fontsize=8)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), 
                 ha='center', va='bottom', fontsize=7, rotation=90)
                 
    plt.tight_layout()
    plt.savefig(save_dir / "map_barchart.png", dpi=300)
    print(f"Đã lưu biểu đồ mAP tại: {save_dir / 'map_barchart.png'}")
    plt.show()

    # --- BIỂU ĐỒ 2: PRECISION-RECALL CURVE ---
    plt.figure(figsize=(12, 8))
    
    # Tạo màu sắc đa dạng cho các đường cong
    colors = plt.cm.jet(np.linspace(0, 1, len(pr_data)))
    
    for idx, (cls_id, data) in enumerate(pr_data.items()):
        prec = data["precision"]
        rec = data["recall"]
        ap = data["ap"]
        
        if ap > 0:
            # Lấy tên răng từ class_names (cls_id bắt đầu từ 1)
            if (cls_id - 1) < len(class_names):
                name = class_names[cls_id - 1]
            else:
                name = f"ID_{cls_id}"
                
            plt.plot(rec, prec, lw=1.5, label=f'{name} (AP={ap:.2f})', color=colors[idx], alpha=0.8)
    
    plt.title('Precision-Recall Curve per Class (Only AP > 0)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", ncol=1, fontsize='small')
    
    plt.tight_layout()
    plt.savefig(save_dir / "pr_curve.png", dpi=300)
    print(f"Đã lưu biểu đồ Precision-Recall tại: {save_dir / 'pr_curve.png'}")
    plt.show()

# --- 3. MAIN ---

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ĐƯỜNG DẪN dỮ LIỆU
    ROOT_DIR = PROJECT_ROOT / "data"
    DIR = ROOT_DIR / "Radiographs"
    ANN = ROOT_DIR / "Segmentation/teeth_polygon.json"
    WEIGHTS_PATH = "data/weights_ETE_train/maskrcnn_epoch67.pth" 

    # Load Data
    md = teeth_dataset.TeethDataset()
    md.load_teeth(DIR, "train", ANN) 
    md.prepare()
    
    dataset = TorchTeethDataset(md, max_size=1333)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Load Model
    num_classes = md.num_classes + 1
    model = maskrcnn_resnet50(pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
    model.to(device)

    # Đánh giá
    mAP, aps, pr_data = evaluate_map(model, data_loader, device, num_classes=num_classes-1, iou_threshold=0.5)
    
    # Vẽ đồ thị
    plot_map_results(aps, pr_data, md.class_names)

    # In kết quả dạng Text
    print(f"\n=== KẾT QUẢ ===")
    print(f"mAP: {mAP:.4f}")

if __name__ == "__main__":
    main()