import math
import torch

class BoxCoder:
    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        """mã hóa các proposals thành các offsets so với ground truth boxes (reference_boxes)"""
        # proposal có shape [N, 4] với N là số bounding boxes
        # Mỗi box được biểu diễn bằng 4 giá trị: [x_min, y_min, x_max, y_max]
        """proposal = tensor([
            [x_min1, y_min1, x_max1, y_max1],  # box 1
            [x_min2, y_min2, x_max2, y_max2],  # box 2
            [x_min3, y_min3, x_max3, y_max3],  # box 3
            ...
        ])"""
        # tính toán chiều rộng, chiều cao và tâm của các hộp đề xuất (proposals) và hộp ground truth (reference_boxes)
        width = proposals[:, 2] - proposals[:, 0]
        height = proposals[:, 3] - proposals[:, 1]
        ctr_x = proposals[:, 0] + 0.5*width
        ctr_y = proposals[:, 1] + 0.5*height

        gt_width = reference_boxes[:, 2] - reference_boxes[:, 0]
        gt_height = reference_boxes[:, 3] - reference_boxes[:, 1]
        gt_ctr_x = reference_boxes[:, 0] + 0.5*width
        gt_ctr_y = reference_boxes[:, 1] + 0.5*height

        """tính offsets"""
        dx = self.weights[0] * (gt_ctr_x - ctr_x) / width
        dy = self.weights[1] * (gt_ctr_y - ctr_y) / height
        dw = self.weights[2] * torch.log(gt_width / width)
        dh = self.weights[3] * torch.log(gt_height / height)

        delta = torch.stack((dx, dy, dw, dh), dim=1)
        return delta
    
    def decode(self, delta, box):
        dx = delta[:, 0] / self.weights[0]
        dy = delta[:, 1] / self.weights[1]
        dw = delta[:, 2] / self.weights[2]
        dh = delta[:, 3] / self.weights[3]

        """giới hạn giá trị của tensor trong một phạm vi từ dw đến bbox_xform_clip"""
        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        width = box[:, 2] - box[:, 0]
        height = box[:, 3] - box[:, 1]
        ctr_x = box[:, 0] + 0.5 * width
        ctr_y = box[:, 1] + 0.5 * height

        # tính toán tọa độ trung tâm, chiều rộng và chiều cao của hộp dự đoán  
        pred_center_x = dx * width + ctr_x
        pred_center_y = dy * height + ctr_y
        pred_width = torch.exp(dw) * width
        pred_height = torch.exp(dh) * height

        """chuyển đổi từ định dạng trung tâm (center format) sang định dạng góc (corner format)"""
        x_min = pred_center_x - 0.5 * pred_width
        y_min = pred_center_y - 0.5 * pred_height
        x_max = pred_center_x + 0.5 * pred_width
        y_max = pred_center_y + 0.5 * pred_height

        target = torch.stack((x_min, y_min, x_max, y_max), dim=1)
        return target
    
def box_iou(box_a, box_b):
    # tính left_top point của vùng giao nhau
    # box_a có shape [N, 4] - N boxes với [x_min, y_min, x_max, y_max]
    left_top = torch.max(box_a[:, None, :2], box_b[:, :2])
    # tính right_bottom point của vùng giao nhau
    right_bottom = torch.min(box_a[:, None, 2:], box_b[:, 2:])
    # tính width và height của vùng giao nhau
    width_height = (right_bottom - left_top).clamp(min=0)
    # tính diện tích vùng giao nhau
    inter = width_height[:, :, 0]*width_height[:, :, 1]
    # tính diện tích của box_a và box_b
    area_a = torch.prod(box_a[:, 2:] - box_a[:, :2], dim=1)
    area_b = torch.prod(box_b[:, 2:] - box_b[:, :2], dim=1)

    iou = inter / (area_a[:, None] + area_b - inter) #overlap over union
    return iou

def process_box(box, score, image_size, min_size):
    """lọc các box nhỏ hơn min_size và cắt các box vượt quá kích thước ảnh"""

    box[:, [0, 2]] = box[:, [0, 2]].clamp(0, max=image_size[1])
    box[:, [1, 3]] = box[:, [1, 3]].clamp(0, max=image_size[0])

    w, h = box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]
    keep = torch.where((w >= min_size) & (h >= min_size))[0]
    box, score = box[keep], score[keep]
    return box, score

def nms(box, score, iou_threshold):
    return torch.ops.torchvision.nms(box, score, iou_threshold)

def batched_nms(box, nms_threshold):
    idx  = torch.arange(box.size[0])
    keep = []

    while idx.size(0) > 0:
        keep.append(idx[0].item()) # Giữ box có score cao nhất
        head_box =  box[idx[0], None, :]
        remain = torch.where(box_iou(head_box, box[idx]) <= nms_threshold)[1]
        idx = idx[remain]

    return keep