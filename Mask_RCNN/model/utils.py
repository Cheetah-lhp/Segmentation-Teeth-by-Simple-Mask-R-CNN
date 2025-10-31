import torch

class Matcher:
    def __init__(self, high_threshold, low_threshold, allow_low_quality_matches=False):
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, iou):
        """
        Arguments:
            iou (Tensor[M, N]): containing the pairwise quality between 
            M ground-truth boxes and N predicted boxes.

        Returns:
            label (Tensor[N]): positive (1) or negative (0) label for each predicted box,
            -1 means ignoring this box.
            matched_idx (Tensor[N]): indices of gt box matched by each predicted box.
        """
        
        value, matched_idx = iou.max(dim=0) # Lấy giá trị iou lớn nhất và index tương ứng cho mỗi predicted box
        label = torch.full((iou.size[1],), -1, dtype=torch.float32, device=iou.device) # khởi tạo label với -1 (ignore)
        
        label[value > self.high_threshold] = 1
        label[value < self.low_threshold] = 0

        if self.allow_low_quality_matches:
            highest_quality = iou.max(dim=1)[0] # Lấy giá trị iou lớn nhất cho mỗi ground truth box
            gt_pred_pairs = torch.where(iou == highest_quality[:, None])[1] # Tìm predicted box tương ứng với mỗi gt box
            label[gt_pred_pairs] = 1

        return label, matched_idx
    
class BalancedPositiveNegativeSampler:
    def __init__(self, num_samples, positive_fraction):
        self.num_samples = num_samples
        self.positive_fraction = positive_fraction # Tỉ lệ positive samples trong tổng số samples

    def __call__(self, labels):
        positive = torch.where(labels == 1)[0]
        negative = torch.where(labels == 0)[0]

        num_pos = int(self.num_samples * self.positive_fraction)
        num_pos = min(positive.numel(), num_pos) # Giới hạn số lượng positive samples không vượt quá số lượng thực tế
        num_neg = self.num_samples - num_pos
        num_neg = min(negative.numel(), num_neg)

        pos_perm = torch.randperm(positive.numel(), device=labels.device)[:num_pos] # Lấy ngẫu nhiên num_pos index từ positive
        neg_perm = torch.randperm(negative.numel(), device=labels.device)[:num_neg]

        pos_idx = positive[pos_perm] # Lấy index của positive và negative samples
        neg_idx = negative[neg_perm]

        return pos_idx, neg_idx
    
def rol_align(feature, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio):
    return torch.ops.torchvision.roi_align(feature, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, False)

class AnchorGenerator:
    def __init__(self, sizes, ratios):
        self.sizes = sizes # Kích thước anchor (32, 64, 128, 256, 512)
        self.ratios = ratios   # Tỉ lệ khung hình anchor (0.5, 1.0, 2.0)
        self.cell_anchor = None
        self._cache = {}

    def set_cell_anchor(self, dtype, device):
        if self.cell_anchor is not None:
            return
        sizes = torch.tensor(self.sizes, dtype=dtype, device=device)
        ratios = torch.tensor(self.ratios, dtype=dtype, device=device)

        h_ratios = torch.sqrt(ratios) # Tính chiều cao dựa trên tỉ lệ và kích thước anchor mặc định
        w_ratios = 1 / h_ratios # tinh chiều rộng dựa trên tỉ lệ và chiều cao anchor 

        hs = (sizes[:, None] * h_ratios[None, :]).view(-1) 
        ws = (sizes[:, None] * w_ratios[None, :]).view(-1) 
        self.cell_anchor = torch.stack([-ws, -hs, ws, hs], dim=1) / 2 # Tạo anchor cho mỗi ô trên feature map và chia đôi để lấy tọa độ từ tâm

    def grid_anchor(self, grid_size, stride):
        dtype, device = self.cell_anchor.dtype, self.cell_anchor.device
        shift_x = torch.arange(0, grid_size[1], dtype=dtype, device=device) * stride[1] # Tạo lưới các điểm tâm anchor trên feature map
        shift_y = torch.arange(0, grid_size[0], dtype=dtype, device=device) * stride[0]
        y, x = torch.meshgrid(shift_y, shift_x) # Tạo lưới 2D từ các điểm tâm
        x = x.reshape(-1)
        y = y.reshape(-1) # Chuyển lưới 2D thành 1D
        shift = torch.stack((x, y, x, y), dim=1).reshape(-1, 1, 4) # Tạo tensor shift để dịch chuyển anchor về đúng vị trí trên feature map
        anchor = (shift + self.cell_anchor).reshape(-1, 4) # Dịch chuyển anchor về đúng vị trí trên feature map
        return anchor
    
    def cached_grid_anchor(self, grid_size, stride):
        key = grid_size + stride 
        if key in self._cache:
            return self._cache[key] # Trả về anchor đã được lưu trong cache nếu có
        anchor = self.grid_anchor(grid_size, stride) # Tạo anchor cho kích thước lưới và stride đã cho
        
        if len(self._cache) >= 3:
            self._cache.clear() # Giới hạn kích thước cache để tránh sử dụng quá nhiều bộ nhớ
        self._cache[key] = anchor # Lưu anchor vào cache
        return anchor
    
    def __call__(self, feature, image_size):
        dtype, device = feature.dtype, feature.device
        grid_size = tuple(feature.shape[-2:]) # Lấy kích thước của feature map
        stride = tuple(int(i/g) for i, g in zip(image_size, grid_size)) # Tính stride dựa trên kích thước ảnh và kích thước feature map

        self.set_cell_anchor(dtype, device) # Thiết lập anchor cho mỗi ô trên feature map
        anchor = self.cached_grid_anchor(grid_size, stride) # Lấy anchor từ cache hoặc tạo mới nếu chưa có
        return anchor