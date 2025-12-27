import albumentations as A

class TeethAugmentor:
    def __init__(self):
        self.rare_ids = []   # Thêm các ID răng sữa A-T vào đây
        
        # 1. Augment NHẸ (Áp dụng cho TẤT CẢ ảnh)
        self.base_transform = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.04, scale_limit=0.04, rotate_limit=8, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.GaussNoise(std_range=(0.02, 0.05), p=0.3),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

        # 2. Augment NẶNG (Chỉ thêm vào khi có răng hiếm)
        self.heavy_transform = A.Compose([
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, p=1.0),
                A.GridDistortion(num_steps=5, distort_limit=0.15, p=1.0),
            ], p=0.3),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    def __call__(self, image, masks, bboxes, labels):
        augmented = self.base_transform(image=image, masks=masks, bboxes=bboxes, class_labels=labels)
        image, masks, bboxes, labels = augmented['image'], augmented['masks'], augmented['bboxes'], augmented['class_labels']

        has_rare = any(lid in self.rare_ids for lid in labels)
        if has_rare:
            heavy_augmented = self.heavy_transform(image=image, masks=masks, bboxes=bboxes, class_labels=labels)
            image, masks, bboxes, labels = heavy_augmented['image'], heavy_augmented['masks'], heavy_augmented['bboxes'], heavy_augmented['class_labels']
            
        return image, masks, bboxes, labels