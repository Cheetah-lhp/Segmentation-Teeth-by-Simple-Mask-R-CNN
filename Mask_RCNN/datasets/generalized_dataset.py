import os #quan ly file
import time # quan ly thoi gian thuc thi
from concurrent.futures import ThreadPoolExecutor, as_completed # thuc thi tac vu song song

import torch
from torchvision import transforms


class GeneralizedDataset:
    
    def __init__(self, max_workers=2, verbose=False):
        self.max_workers = max_workers
        self.verbose = verbose
            
    def __getitem__(self, i):
        img_id = self.ids[i]
        image = self.get_image(img_id)
        image = transforms.ToTensor()(image)
        target = self.get_target(img_id) if self.train else {}
        return image, target   
    """truy cap va kiem tra mot (mau~) phan tu trong dataset tai vi tri i"""
    def __len__(self):
        return len(self.ids)
    """so luong mau du lieu trong dataset"""

    def check_dataset(self, checked_id_file):
        """
        use multithreads to accelerate the process.
        check the dataset to avoid some problems listed in method `_check`.
        """
        
        if os.path.exists(checked_id_file):
            info = [line.strip().split(", ") for line in open(checked_id_file)]
            self.ids, self.aspect_ratios = zip(*info)
            return
        """kiem tra neu da co file kiem tra truoc do"""

        since = time.time()
        print("Checking the dataset...")
        
        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        seqs = torch.arange(len(self)).chunk(self.max_workers)
        tasks = [executor.submit(self._check, seq.tolist()) for seq in seqs]
        """ chia du lieu thanh nhieu phan (chunk) tuong ung so luong, moi luong goi ham _check() de kiem tra du lieu"""
        
        outs = []
        for future in as_completed(tasks):
            outs.extend(future.result()) #cac anh hop le duoc dua vao outs
        if not hasattr(self, "id_compare_fn"):
            self.id_compare_fn = lambda x: int(x)
        outs.sort(key=lambda x: self.id_compare_fn(x[0])) # sap xep danh sach theo id anh
        
        with open(checked_id_file, "w") as f:
            for img_id, aspect_ratio in outs:
                f.write("{}, {:.4f}\n".format(img_id, aspect_ratio))
        # ghi ra file ket qua, moi dong: image_id, aspect_ratio
        info = [line.strip().split(", ") for line in open(checked_id_file)]
        self.ids, self.aspect_ratios = zip(*info)
        print("checked id file: {}".format(checked_id_file))
        print("{} samples are OK; {:.1f} seconds".format(len(self), time.time() - since))
        
    def _check(self, seq):
        out = []
        for i in seq:
            img_id = self.ids[i]
            target = self.get_target(img_id)
            boxes = target["boxes"]
            labels = target["labels"]
            masks = target["masks"]

            try:
                assert len(boxes) > 0, "{}: len(boxes) = 0".format(i)
                assert len(boxes) == len(labels), "{}: len(boxes) != len(labels)".format(i)
                assert len(boxes) == len(masks), "{}: len(boxes) != len(masks)".format(i)

                out.append((img_id, self._aspect_ratios[i]))
            except AssertionError as e:
                if self.verbose:
                    print(img_id, e)
        return out
    """check 3 dieu kien:
        1. anh phai co it nhat 1 bbox
        2. so luong boxes = so luong lables
        3. so luong boxes = so luong masks
        => neu hop le them vao out img_id va aspect_ratios
        => neu k hop le in ra loi (verbose=true)"""
                    