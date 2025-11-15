import matplotlib.pyplot as plt
from Mask_RCNN.datasets.dataset import DentalDataset

dataset = DentalDataset(root_dir=r"C:\Users\Admin\OneDrive\Dokumen\AIOT Lab\My Weekly Report\Teeth Segmentation with Mask R-CNN\Tuft Dental Database", train=True)
#lay mau dau tien
image, target = dataset[0]
print("Image shape:", image.shape)
print("Target keys:", target.keys())
print("Number of objects:", len(target["boxes"]))
#hien thi anh + mask dau tien
plt.figure(figsize=(6,6))
plt.imshow(image.permute(1,2,0))

mask = target["masks"][0].numpy() 
plt.imshow(mask, alpha=0.5, cmap='Reds')
plt.show()

