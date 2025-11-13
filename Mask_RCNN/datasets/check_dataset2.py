import matplotlib.pyplot as plt
import numpy as np
from dataset import DentalDataset

dataset = DentalDataset(root_dir=r"C:\Users\Admin\OneDrive\Dokumen\AIOT Lab\My Weekly Report\Teeth Segmentation with Mask R-CNN\Tuft Dental Database", train=True)

image, target = dataset[0]
img_np = image.permute(1,2,0).numpy()

plt.figure(figsize=(8,8))
plt.imshow(img_np)

num_objects = target["masks"].shape[0]

# Dùng colormap 'hsv' chung
cmap = plt.get_cmap('hsv')

for i in range(min(10, num_objects)): #chi hien thi 10 mask dau, qua nhieu se khong hien thi het duoc
    mask = target["masks"][i].numpy()
    # Chuyển mask 0/1 sang float để colormap nhận
    plt.imshow(mask, alpha=0.5, cmap='hsv')
    #plt.imshow(mask * (i+1)/num_objects, alpha=0.4, cmap=cmap)

plt.title(f"Image ID: {target['image_id'].item()}, {num_objects} objects")
plt.axis('off')
plt.show()
