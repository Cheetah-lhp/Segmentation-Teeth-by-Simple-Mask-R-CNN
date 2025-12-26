import albumentations as A
<<<<<<< HEAD

class TeethAugmentor:
    def __init__(self):
        self.rare_ids = [1, 16] + list(range(33, 53)) # Thêm các ID răng sữa A-T vào đây
        
        # 1. Augment NHẸ (Áp dụng cho TẤT CẢ ảnh)
        self.base_transform = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

        # 2. Augment NẶNG (Chỉ thêm vào khi có răng hiếm)
        self.heavy_transform = A.Compose([
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=30, p=1.0),
=======
import cv2
import random

class TeethAugmentor:
    def __init__(self):
        self.rare_ids = [1, 16] + list(range(33, 53)) 
        
        # 1. Augment NHẸ
        self.base_transform = A.Compose([
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.95, 1.05),
                rotate=(-10, 10),
                p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.GaussNoise(std_range=(0.02, 0.07), p=0.3),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

        # 2. Augment NẶNG
        self.heavy_transform = A.Compose([
            A.OneOf([
                A.ElasticTransform(alpha=1, sigma=50, p=1.0),
>>>>>>> 766571ee5d967372d828ae5b12785cdf77311d99
                A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0),
            ], p=1.0),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    def __call__(self, image, masks, bboxes, labels):
<<<<<<< HEAD
        # Luôn áp dụng augment nhẹ cho tất cả
        augmented = self.base_transform(image=image, masks=masks, bboxes=bboxes, class_labels=labels)
        image, masks, bboxes, labels = augmented['image'], augmented['masks'], augmented['bboxes'], augmented['class_labels']

        # Nếu có răng hiếm, bồi thêm augment nặng
=======
        augmented = self.base_transform(image=image, masks=masks, bboxes=bboxes, class_labels=labels)
        image, masks, bboxes, labels = augmented['image'], augmented['masks'], augmented['bboxes'], augmented['class_labels']

>>>>>>> 766571ee5d967372d828ae5b12785cdf77311d99
        has_rare = any(lid in self.rare_ids for lid in labels)
        if has_rare:
            heavy_augmented = self.heavy_transform(image=image, masks=masks, bboxes=bboxes, class_labels=labels)
            image, masks, bboxes, labels = heavy_augmented['image'], heavy_augmented['masks'], heavy_augmented['bboxes'], heavy_augmented['class_labels']
            
        return image, masks, bboxes, labels