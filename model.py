## Siamese network model

import torch
import torch.nn as nn
import torch.nn.functional as F



class Siamese(nn.Module):

    def __init__(self,in_channels):
        self.in_c = in_channels
        super(Siamese,self).__init__()

        ## change to custom architecture.. with similar progression as in siamese paper.
        ## Assuming input image size as 3 * 224 * 224 or 1 * 224 * 224
        ## layer definitions
        self.conv1 = nn.Conv2d(self.in_c,32,kernel_size=13) ## 212
        self.pool = nn.MaxPool2d(kernel_size=2) ## 106
        self.conv2 = nn.Conv2d(32,64,kernel_size=9) ## 98
        ## pool-->49
        self.conv3  = nn.Conv2d(64,128,kernel_size=7) ## 43
        ## pool--> 21
        self.conv4 = nn.Conv2d(128,256,kernel_size=5) ## 17
        ## pool---> 8
        self.conv5 = nn.Conv2d(256,512,kernel_size=3) ## 6
        self.fc1 = nn.Linear(18432,4096)
        self.fc2 = nn.Linear(4096,1)

    ## for one image
    def _forward(self,x):
        for conv in [self.conv1,self.conv2,self.conv3,self.conv4]:
            x = conv(x)
            x = F.relu(x)
            x = self.pool(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = x.view(x.shape[0],-1)
        # print(x.shape)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x

    ## REF: https://github.com/kevinzakka/one-shot-siamese/blob/master/model.py
    def forward(self,im1,im2):

        im1_encoding = self._forward(im1)
        im2_encoding = self._forward(im2)

        dist = torch.abs(im1_encoding-im2_encoding)

        scores = self.fc2(dist)

        # p = F.sigmoid(self.fc2(dist))  ## will be using BCEWithLogitsLoss for stability
        return scores

if __name__== "__main__":

    model = Siamese(3)

    im1 = torch.randn((1,3,224,224))
    im2 = torch.randn((1,3, 224, 224))

    score = model(im1,im2)
    print(score.shape)
