import torch
import math

from .utils import roi_align

class RoIAlign:
    def __init__(self, output_size, sampling_ratio):
        self.output_size = output_size
        self.sampling_ratio = sampling_ratio

    def __call__(self, features, proposals, image_shape):
        # Level mapping logic
        w = proposals[:, 2] - proposals[:, 0]
        h = proposals[:, 3] - proposals[:, 1]
        s = torch.sqrt(w * h)
        # Công thức chuẩn FPN: k = floor(4 + log2(sqrt(area)/224))
        target_lvls = torch.floor(4 + torch.log2(s / 224 + 1e-6))
        target_lvls = torch.clamp(target_lvls, min=2, max=5).to(torch.int64) - 2
        
        output = torch.zeros((proposals.shape[0], 256, self.output_size[0], self.output_size[1]), device=proposals.device)
        
        for level, (name, feature) in enumerate(features.items()):
            if level > 3: break # Chỉ dùng P2-P5 cho RoIAlign (P6 chỉ dành cho RPN)
            idx = torch.where(target_lvls == level)[0]
            if len(idx) == 0: continue
                
            rois = proposals[idx]
            batch_idx = torch.zeros((rois.shape[0], 1), device=proposals.device)
            rois_final = torch.cat([batch_idx, rois], dim=1)
            scale = 1.0 / (4 * (2**level))
            
            output[idx] = roi_align(feature, rois_final, scale, self.output_size[0], self.output_size[1], self.sampling_ratio)
            
        return output
    
    """    
    input class RoIAlign:
    + feature: Tensor [N, C, H, W] - Feature maps
    + proposal: Tensor [K, 4] - Proposal boxes (x1, y1, x2, y2)
    + image_shape: [H, W] - Kích thước ảnh gốc

    output:
    tensor[K, C, output_height, output_width] - Cac vung dac trung da duoc align
    """