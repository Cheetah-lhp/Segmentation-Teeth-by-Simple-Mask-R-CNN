import torch
import os
import csv
import time
from torch.utils.data import DataLoader
from Mask_RCNN.model.mask_rcnn import maskrcnn_resnet50
from Mask_RCNN.dataset import TeethDataset, TorchTeethDataset
import torch.optim as optim

def collate_fn(batch):
    # Lọc bỏ những sample mà target không có boxes (răng)
    batch = [b for b in batch if b is not None and b[1]["boxes"].numel() > 0]
    if len(batch) == 0:
        return None # Trả về None nếu cả batch toàn ảnh trống
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
#-------------------------train_epoch-----------------------------------------------------------------------------
def train_one_epoch(model, optimizer, data_loader, device):
    model.train()

    loss_sums = {
        "roi_classifier_loss": 0.0,
        "roi_box_loss": 0.0,
        "roi_mask_loss": 0.0,
        "rpn_objectness_loss": 0.0,
        "rpn_box_loss": 0.0,
        "total_loss": 0.0
    }

    num_batches = 0

    for data in data_loader:
        if data is None:
            continue

        images, targets = data
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        for k in loss_dict:
            loss_sums[k] += loss_dict[k].item()

        loss_sums["total_loss"] += total_loss.item()
        num_batches += 1

    for k in loss_sums:
        loss_sums[k] /= num_batches

    return loss_sums

#-----------------------valid_epoch----------------------------------------------------------------------------------
@torch.no_grad()
def valid_one_epoch(model, data_loader, device):
    model.train()

    loss_sums = {
        "roi_classifier_loss": 0.0,
        "roi_box_loss": 0.0,
        "roi_mask_loss": 0.0,
        "rpn_objectness_loss": 0.0,
        "rpn_box_loss": 0.0,
        "total_loss": 0.0
    }

    num_batches = 0

    for data in data_loader:
        if data is None:
            continue

        images, targets = data
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        total_loss = sum(loss for loss in loss_dict.values())

        for k in loss_dict:
            loss_sums[k] += loss_dict[k].item()
        loss_sums["total_loss"] += total_loss.item()
        num_batches += 1

    for k in loss_sums:
        loss_sums[k] /= max(num_batches, 1)

    return loss_sums

#----------------main-------------------------------------------------------------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ROOT_DIR = os.path.abspath("./")
    DIR = os.path.join(ROOT_DIR, "data/Radiographs")
    ANNOTATION_DIR = os.path.join(ROOT_DIR, "data/Segmentation/teeth_polygon.json")

    md_train = TeethDataset()
    md_train.load_teeth(DIR, "train", ANNOTATION_DIR)
    md_train.prepare()
    train_set = TorchTeethDataset(md_train, max_size=1333)

    md_val = TeethDataset()
    md_val.load_teeth(DIR, "val", ANNOTATION_DIR)
    md_val.prepare()
    val_set = TorchTeethDataset(md_val, max_size=1333)


    """batch_size: so luong sample (anh) duoc dua vao model trong 1 lan forward+backward"""
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_fn)

    """so label + 1 background"""
<<<<<<< HEAD
    num_classes = md_train.num_classes + 1
    model = maskrcnn_resnet50(pretrained=True, num_classes=num_classes) 
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    warmup_epochs = 5
=======
    num_classes = md.num_classes + 1
    model = maskrcnn_resnet50(pretrained=True, num_classes=num_classes) 

    model.to(device)
>>>>>>> 766571ee5d967372d828ae5b12785cdf77311d99
    num_epochs = 60
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # Tăng dần LR trong 5 epoch đầu để ổn định mô hình
    warmup_epochs = 5
    lr_start = 1e-4

    # 1. Scheduler tăng dần (Warmup)
    warmup_sch = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    # 2. Scheduler giảm dần (Cosine)
    cosine_sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(num_epochs - warmup_epochs), eta_min=1e-6)
    # Kết hợp cả 2
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_sch, cosine_sch], milestones=[warmup_epochs])
<<<<<<< HEAD

    LOG_DIR = os.path.join(ROOT_DIR, "logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FILE = os.path.join(LOG_DIR, "train_log.csv")

    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "lr",

                "train_total_loss",
                "val_total_loss",

                "best_val_loss",
                "best_epoch",

                "train_roi_classifier_loss",
                "train_roi_box_loss",
                "train_roi_mask_loss",
                "train_rpn_objectness_loss",
                "train_rpn_box_loss",

                "val_roi_classifier_loss",
                "val_roi_box_loss",
                "val_roi_mask_loss",
                "val_rpn_objectness_loss",
                "val_rpn_box_loss",
                
                "is_best",
                "epoch_time_sec"
            ])
        #####phuc
    best_val_loss = float("inf")
    best_epoch = -1

    #--------------------------------------------epoch---------------------------------------------------------------
    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train_one_epoch(model, optimizer, train_loader, device)
        val_loss = valid_one_epoch(model, val_loader, device)
        
        scheduler.step() 
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - start_time

        is_best = int(val_loss["total_loss"] < best_val_loss)
        if is_best:
            best_val_loss = val_loss["total_loss"]
            best_epoch = epoch+1

        print(f"Epoch {epoch+1}/{num_epochs}, Train_loss: {train_loss["total_loss"]:.4f}, Val_loss: {val_loss["total_loss"]:.4f}, LR: {current_lr:.6f}")

        #######checkpoint
        SAVE_DIR = os.path.join(ROOT_DIR, "data/checkpoints")
        os.makedirs(SAVE_DIR, exist_ok=True)

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": train_loss,
            "num_classes": num_classes,
            "best_val_loss": best_val_loss,
        }
        if is_best:
            torch.save(checkpoint, os.path.join(SAVE_DIR, "best.pth"))
        torch.save(
            checkpoint,
            os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch+1}.pth")
        )
        #####checkpoint

        ####csv
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                current_lr,

                train_loss["total_loss"],
                val_loss["total_loss"],

                best_val_loss,
                best_epoch,

                train_loss["roi_classifier_loss"],
                train_loss["roi_box_loss"],
                train_loss["roi_mask_loss"],
                train_loss["rpn_objectness_loss"],
                train_loss["rpn_box_loss"],

                val_loss["roi_classifier_loss"],
                val_loss["roi_box_loss"],
                val_loss["roi_mask_loss"],
                val_loss["rpn_objectness_loss"],
                val_loss["rpn_box_loss"],

                is_best,
                epoch_time
            ])
        ####csv
=======

    #Train trên sever
    # LOG_DIR = os.path.join(ROOT_DIR, "logs")
    # os.makedirs(LOG_DIR, exist_ok=True)
    # LOG_FILE = os.path.join(LOG_DIR, "train_log.csv")

    # if not os.path.exists(LOG_FILE):
    #     with open(LOG_FILE, "w", newline="") as f:
    #         writer = csv.writer(f)
    #         writer.writerow([
    #             "epoch",
    #             "total_loss",
    #             "roi_classifier_loss",
    #             "roi_box_loss",
    #             "roi_mask_loss",
    #             "rpn_objectness_loss",
    #             "rpn_box_loss",
    #             "learning_rate",
    #             "is_best",
    #             "epoch_time_sec"
    #         ])
    #     #####phuc
    # best_loss = float("inf")

    # for epoch in range(num_epochs):
    #     start_time = time.time()

    #     loss = train_one_epoch(model, optimizer, train_loader, device)
    #     scheduler.step() 
    #     current_lr = optimizer.param_groups[0]['lr']
    #     epoch_time = time.time() - start_time

    #     is_best = loss["total_loss"] < best_loss
    #     if is_best:
    #         best_loss = loss["total_loss"]

    #     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss["total_loss"]:.4f}, LR: {current_lr}")

    #     #######checkpoint
    #     SAVE_DIR = os.path.join(ROOT_DIR, "data/checkpoints")
    #     os.makedirs(SAVE_DIR, exist_ok=True)

    #     checkpoint = {
    #         "epoch": epoch + 1,
    #         "model_state_dict": model.state_dict(),
    #         "optimizer_state_dict": optimizer.state_dict(),
    #         "scheduler_state_dict": scheduler.state_dict(),
    #         "loss": loss,
    #         "num_classes": num_classes,
    #         "best_loss": best_loss,
    #     }
    #     if is_best:
    #         torch.save(checkpoint, os.path.join(SAVE_DIR, "best.pth"))
    #     torch.save(
    #         checkpoint,
    #         os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch+1}.pth")
    #     )
    #     #####checkpoint

    #     ####csv
    #     with open(LOG_FILE, "a", newline="") as f:
    #         writer = csv.writer(f)
    #         writer.writerow([
    #             epoch + 1,
    #             loss["total_loss"],
    #             loss["roi_classifier_loss"],
    #             loss["roi_box_loss"],
    #             loss["roi_mask_loss"],
    #             loss["rpn_objectness_loss"],
    #             loss["rpn_box_loss"],
    #             current_lr,
    #             best_loss,
    #             epoch_time
    #         ])
    #     ####csv
    #     SAVE_DIR = os.path.join(ROOT_DIR, "data/weights_ETE_train")
    #     os.makedirs(SAVE_DIR, exist_ok=True)
    #     torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"maskrcnn_epoch{epoch+1}.pth"))
    #     #free VRAM moi epoch
    #     torch.cuda.empty_cache()

    # Train trên máy local
    for epoch in range(num_epochs):
        loss = train_one_epoch(model, optimizer, train_loader, device)
        scheduler.step() 
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss["total_loss"]:.4f}, LR: {current_lr}")

>>>>>>> 766571ee5d967372d828ae5b12785cdf77311d99
        SAVE_DIR = os.path.join(ROOT_DIR, "data/weights_ETE_train")
        os.makedirs(SAVE_DIR, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"maskrcnn_epoch{epoch+1}.pth"))
        #free VRAM moi epoch
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()