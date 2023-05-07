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


def pic_preprocess(raw_pics, pics, size):
    i = 0
    for filename in tqdm(os.listdir(raw_pics)):
        img = cv2.imread(raw_pics + '/' + filename)
        img = cv2.resize(img, size)
        cv2.imwrite(pics + '/' + str(i) + ".jpg", img)
        i += 1


if __name__ == '__main__':
    pic_preprocess("raw_pics", "pics", (64, 64))


