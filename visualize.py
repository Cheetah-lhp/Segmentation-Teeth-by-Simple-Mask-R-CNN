import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

def visualize_prediction(image_tensor, preds, score_thresh, topk, min_area = 3000):
    """
    image_tensor: dang (3, H, W)
    preds: dictionary của Mask R-CNN: boxes, labels, masks, scores
    """

    image = image_tensor.permute(1, 2, 0).cpu().numpy().copy()
    image = (image * 255).astype(np.uint8)

    boxes = preds["boxes"].cpu().numpy()
    scores = preds["scores"].cpu().numpy()
    masks = preds["masks"].cpu().numpy()[:, 0]   # (N, 1, H, W)

    idx = np.where(scores >= score_thresh)[0]
    boxes = boxes[idx]
    scores = scores[idx]
    masks = masks[idx]

    if len(scores) > topk:
        top_idx = np.argsort(scores)[-topk:]
        boxes = boxes[top_idx]
        scores = scores[top_idx]
        masks = masks[top_idx]

    for i in range(len(boxes)):
        box = boxes[i].astype(int)
        mask = masks[i]

        mask_bin = (mask > 0.5).astype(np.uint8)

        # if mask_bin.sum() < min_area:
        #     continue

        # random mau cho moi rang
        color = np.array([random.randint(0,255), random.randint(0, 255), random.randint(0, 255)])
        
        colored_mask = np.zeros_like(image)
        for c in range(3):
            colored_mask[:, :, c] = mask_bin * color[c]

        image = cv2.addWeighted(image, 1.0, colored_mask, 0.4, 0)

        # Draw bounding box
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color.tolist(), 2)

        cv2.putText(image, f"{scores[i]:.2f}", (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color.tolist(), 2)

    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.axis("off")
    plt.show()
