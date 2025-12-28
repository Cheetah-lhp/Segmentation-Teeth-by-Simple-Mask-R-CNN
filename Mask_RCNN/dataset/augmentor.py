import albumentations as A

class TeethAugmentor:
    def __init__(self):        
        self.transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.15,contrast_limit=0.15,p=0.8),
            A.GaussNoise( std_range=(0.01, 0.03), p=0.2),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.03, rotate_limit=5, p=0.3),
            A.OneOf([
                A.ElasticTransform(alpha=0.5, sigma=40, p=1.0),
                A.GridDistortion(num_steps=5, distort_limit=0.1, p=1.0),
            ], p=0.2),],
            bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels'],min_visibility=0.3))

    def __call__(self, image, masks, bboxes, labels):
        try:
            augmented = self.transform(image=image, masks=masks, bboxes=bboxes, class_labels=labels)
            return augmented['image'], augmented['masks'], augmented['bboxes'], augmented['class_labels']
        except Exception as e:
            return image, masks, bboxes, labels