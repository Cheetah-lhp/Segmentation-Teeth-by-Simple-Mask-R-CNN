import albumentations as A
import cv2

class RareTeethAugmentor:
    def __init__(self):
        self.rare_ids = list(range(33, 53))  # ID của các răng hiếm (răng khôn, răng sữa, răng phụ)
        
        # Bộ Augment nặng (Chỉ dùng cho răng hiếm)
        self.heavy_transform = A.Compose([
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
            ], p=1.0),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(p=0.3)
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    def __call__(self, image, masks, bboxes, labels):
        # Kiểm tra xem trong các label có cái nào thuộc danh sách hiếm không
        has_rare = any(lid in self.rare_ids for lid in labels)
        
        if has_rare:
            # Nếu có răng hiếm -> Chạy biến dạng nặng
            augmented = self.heavy_transform(image=image, masks=masks, bboxes=bboxes, class_labels=labels)
            return augmented['image'], augmented['masks'], augmented['bboxes'], augmented['class_labels']
        else:
            # Nếu không có -> Trả về nguyên bản (Tiết kiệm CPU/RAM)
            return image, masks, bboxes, labels