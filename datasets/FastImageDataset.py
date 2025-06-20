import os
from PIL import Image

from torch.utils.data import Dataset
import cv2


class FastImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_test=False):
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        self.samples = []

        if is_test:
            self.samples = [os.path.join(root_dir, fname)
                            for fname in sorted(os.listdir(root_dir)) if fname.lower().endswith('.jpg')]
        else:
            self.classes = sorted(os.listdir(root_dir))
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

            for cls_name in self.classes:
                cls_folder = os.path.join(root_dir, cls_name)
                self.samples.extend([(os.path.join(cls_folder, fname), self.class_to_idx[cls_name])
                                     for fname in os.listdir(cls_folder) if fname.lower().endswith('.jpg')])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.is_test:
            img_path = self.samples[idx]
            image = cv2.imread(img_path)  # OpenCV 활용하여 빠른 이미지 로딩
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB 변환
            image = Image.fromarray(image)
            
            if self.transform:
                image = self.transform(image)
            return image
        else:
            img_path, label = self.samples[idx]
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

            if self.transform:
                image = self.transform(image)
            return image, label