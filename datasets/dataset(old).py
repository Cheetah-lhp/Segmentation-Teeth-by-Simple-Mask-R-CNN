import torch
import glob
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class DentalDataset(Dataset):
    def __init__(self, img_dirs, mask_dirs, transform_img=None, transform_mask=None):
        self.img_dirs = sorted(glob.glob(os.path.join(img_dirs, '*.jpg'))) #glob: dùng để tự động tìm tất cả các file khớp với mẫu (pattern) như "*.png", thay vì phải liệt kê thủ công.sorted() để đảm bảo thứ tự ảnh và mask trùng nhau
        self.mask_dirs = sorted(glob.glob(os.path.join(mask_dirs, '*.jpg')))
        self.transform_img = transform_img
        self.transform_mask = transform_mask

    def __len__(self):
        return len(self.img_dirs)
    
    def __getitem__(self, index):
        img = Image.open(self.img_dirs[index]).convert('L') #convert('L') để chuyển ảnh về dạng grayscale
        mask = Image.open(self.mask_dirs[index]).convert('L')

        if self.transform_img:
            img = self.transform_img(img)
        
        if self.transform_mask:
            mask = self.transform_mask(mask)
        
        return img, mask
    
    def get_dataloaders(img_dirs, mask_dirs, transform_img=None, transform_mask=None):
        dataset = DentalDataset(img_dirs, mask_dirs, transform_img, transform_mask)
        train_size, val_size, test_size = 760, 190, 70
        train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

        train_loader = DataLoader(train_set, batch_size=10, shuffle=False, num_workers=2)
        val_loader = DataLoader(val_set, batch_size=10, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_set, batch_size=10,shuffle=False, num_workers=2)

        return train_loader, val_loader, test_loader   
        