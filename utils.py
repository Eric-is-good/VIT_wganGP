import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import os

from tqdm import tqdm


class MyDataSet(Dataset):
    def __init__(self, images_path):
        self.imglist = os.listdir(images_path)
        self.images = []
        for img in tqdm(self.imglist):
            image = cv2.imread(os.path.join(images_path, img))
            image = image.transpose(2, 0, 1)
            image = torch.from_numpy(image).float() / 255.0
            self.images.append(image)

    def __getitem__(self, index):
        return self.images[index]

    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        return images


def read_path(file_pathname, small_path):
    i = 0
    for filename in tqdm(os.listdir(file_pathname)):
        img = cv2.imread(file_pathname + '/' + filename)
        img = cv2.resize(img, (64, 64))
        cv2.imwrite(small_path + '/' + str(i) + ".jpg", img)
        i += 1


if __name__ == '__main__':
    read_path("data","your_path")


