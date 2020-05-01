import os
import torch
import pandas as pd
from skimage import io, transform, color
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets

# np.random.seed(10)


transform_ = transforms.Compose([
    # transforms.CenterCrop(250),
    # transforms.RandomResizedCrop(224, (0.8, 1.0)),
    # transforms.RandomHorizontalFlip(),
    transforms.Resize((224,224)),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

## with p=0.5 picking 2 images from a random class and with p_=0.5 picking 2 images from two different classes.
## Otherway is to geneate a csv doing above ^^ with file names in it and just load the images using the filenames here.
class SiameseDataset(Dataset):

    def __init__(self,dir,transform,len, p, batch_size):
        self.dir = dir
        self.transform = transform
        self.p = p
        self.len = len

        self.label = 0
        self.count = 0
        self.bs = batch_size

    def __len__(self):
        return self.len ## determines no of steps of epoch = len/batch size, usually equals to len(train_images).

    def __getitem__(self, item):

        # print(self.label)
        self.count += 1
        if self.count == self.bs :
            self.label = (self.label + 1)%200
            self.count = 0

        p = np.random.uniform(0,1) ## to reduce overfitting. each 64 sized batch contains 4 same, 60 different.
        different = 1
        if p<self.p:
            ## pick same class
            different = 0
            # label = np.random.randint(0,200)
            label = self.label
            labels = [label,label]
            label_dir = os.path.join(self.dir,str(label))
            img_files = os.listdir(label_dir)
            img_ids = np.random.choice(img_files,size=2,replace=False)
            imgs = [Image.open(os.path.join(self.dir,label_dir,im)) for im in img_ids]
            imgs = [im.convert('RGB') for im in imgs]
            imgs = [self.transform(im) for im in imgs]

        else:
            ## pick different class
            label = np.random.choice([i for i in range(200) if i!=self.label],size=2,replace=False)
            labels = [self.label,label[0]]
            imgs = []
            for label in labels:
                label_dir = os.path.join(self.dir, str(label))
                img_files = os.listdir(label_dir)
                img_id = np.random.choice(img_files)
                img = Image.open(os.path.join(self.dir,label_dir,img_id))
                img = img.convert('RGB')
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
    d = SiameseDataset('/Users/sai/Downloads/ssl_trainval_data/train', transform_,10,0.5,2)
    dataloader = DataLoader(d, batch_size=2, shuffle=True)
    for i,sample_batch in enumerate(dataloader):
        visualize(sample_batch,2)
    for i,sample_batch in enumerate(dataloader):
        visualize(sample_batch,2)