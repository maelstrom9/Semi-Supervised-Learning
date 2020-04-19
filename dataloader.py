import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets

np.random.seed(10)

transform_ = transforms.Compose([
    # you can add other transformations in this list
    transforms.ToPILImage(),
    transforms.CenterCrop(300),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

## with p=0.5 picking 2 images from a random class and with p_=0.5 picking 2 images from two different classes.
## Otherway is to geneate a csv doing above ^^ with file names in it and just load the images using the filenames here.
class SiameseDataset(Dataset):

    def __init__(self,dir,transform):
        self.dir = dir
        self.transform = transform

    def __len__(self):
        return 10 ## determines no of steps of epoch, usually equals to len(train_images).

    def __getitem__(self, item):

        p = np.random.randint(0,2)
        different = 1
        if p:
            ## pick same class
            different = 0
            label = np.random.randint(0,200)
            labels = [label,label]
            label_dir = os.path.join(self.dir,str(label))
            img_files = os.listdir(label_dir)
            img_ids = np.random.choice(img_files,size=2,replace=False)

            imgs = [io.imread(os.path.join(self.dir,label_dir,im)) for im in img_ids]
            imgs = [self.transform(im) for im in imgs]

        else:
            ## pick different class
            labels = np.random.choice([i for i in range(200)],size=2,replace=False)
            imgs = []
            for label in labels:
                label_dir = os.path.join(self.dir, str(label))
                img_files = os.listdir(label_dir)
                img_id = np.random.choice(img_files)
                img = io.imread(os.path.join(self.dir, label_dir, img_id))
                img = self.transform(img)
                imgs.append(img)

        return {'Image1': imgs[0], 'Image2': imgs[1], "target": torch.tensor(different),
                "im1_label": torch.tensor(labels[0]), "im2_label": torch.tensor(labels[1])}


def visualize(sample,batch_size):

    # print(sample['Image1'].shape, sample['target'],sample['im1_label'],sample['im2_label'])
    image1 = sample['Image1']
    image2 = sample['Image2']
    target = sample['target']
    im1_label = sample['im1_label']
    im2_label = sample['im2_label']

    for i in range(batch_size):
        im1 = image1[i]
        im2 = image2[i]

        im1 = im1.squeeze(0).permute(1, 2, 0).numpy()
        im2 = im2.squeeze(0).permute(1, 2, 0).numpy()

        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(im1)
        axarr[1].imshow(im2)

        plt.title("{}".format(target[i]))
        plt.pause(2)




if __name__ == "__main__":
    d = SiameseDataset('/Users/sai/Downloads/ssl_trainval_data/train', transform_)
    dataloader = DataLoader(d, batch_size=2, shuffle=True)
    for i,sample_batch in enumerate(dataloader):
        visualize(sample_batch,2)
    for i,sample_batch in enumerate(dataloader):
        visualize(sample_batch,2)