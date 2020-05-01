
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils, datasets, models
import matplotlib.pyplot as plt

from resnet_model import ResnetSiamese,Identity
from dataloader import SiameseDataset

## datatranforms
RandomTransforms = [ transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),transforms.RandomPerspective(0.2)]
data_transforms = {
    "train" : transforms.Compose([
        transforms.Resize((260,260)),
        transforms.RandomResizedCrop(224,(0.8, 1.0)),
        transforms.RandomApply(RandomTransforms),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
     "val" : transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((260,260)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
}

resn = models.resnet50(pretrained=True)
resn.fc = Identity()
model = ResnetSiamese(3, resn)
model.load_state_dict(torch.load("best_model.pt", map_location=torch.device('cpu'))
)
# resn =  models.resnet50(pretrained=True)
# model = Siamese(3)

# path = "/Users/sai/Downloads/semi-inat-2020/test/test"
path = "/Users/sai/Downloads/ssl_trainval_data/val"
t_path = "/Users/sai/Downloads/ssl_trainval_data/train"

def visualize():
    img1 = Image.open(os.path.join(t_path,"0","0.jpg"))
    img1 = data_transforms['train'](img1)
    img1 = img1.unsqueeze(0)
    for img_name in os.listdir(os.path.join(t_path,"1")):
        img2 = Image.open(os.path.join(t_path,"1",img_name))
        img2 = data_transforms['train'](img2)
        img2 = img2.unsqueeze(0)

        score = model(img1,img2)
        prob = torch.sigmoid(score)
        print(prob)

        im1 = img1.squeeze(0).permute(1, 2, 0).numpy()
        im2 = img2.squeeze(0).permute(1, 2, 0).numpy()

        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(im1)
        axarr[1].imshow(im2)

        res = 1 if prob>0.5 else 0
        plt.title("{}".format(res))
        plt.pause(2)

visualize()

# score = model(im1,im2)
# print(score)