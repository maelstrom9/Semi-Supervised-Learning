## Resent-Siamese network model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from attn_model.residual_attention_network import ResidualAttentionModel_92


'''Using Resnet Backbone in the Siamese'''

class ResnetSiamese(nn.Module):

    def __init__(self,in_channels,model):
        self.in_c = in_channels
        super(ResnetSiamese,self).__init__()

        self.model = model
        self.fc = nn.Linear(2048,1)

    ## for one image
    def _forward(self,x):
        return self.model(x)

    ## REF: https://github.com/kevinzakka/one-shot-siamese/blob/master/model.py
    def forward(self,im1,im2):

        im1_encoding = self._forward(im1)
        im2_encoding = self._forward(im2)

        # print(im2_encoding)

        dist = torch.abs(im1_encoding-im2_encoding)

        scores = self.fc(dist)

        # p = F.sigmoid(self.fc2(dist))  ## will be using BCEWithLogitsLoss for stability
        return scores

class Identity(nn.Module):
    def __init__(self,):
        super(Identity,self).__init__()

    def forward(self,x):
        return x


if __name__== "__main__":

    # resn = models.resnet18(pretrained=True)
#     resn = models.resnet50(pretrained=True)
#     print(resn)
#     num_ftrs = resn.fc.in_features
#     resn.fc = nn.Linear(num_ftrs, 200)
#     resn.load_state_dict(torch.load("ssl_best_model.pt", map_location=torch.device('cpu'))
# )
#     # resn =  models.resnet50(pretrained=True)
#     # model = Siamese(3)
#     resn.fc = Identity()

    resn = ResidualAttentionModel_92()
    resn.fc = Identity()

    model = ResnetSiamese(3, resn)

    im1 = torch.randn((2,3,224,224))
    im2 = torch.randn((2,3,224,224))

    score = model(im1,im2)
    print(score.shape)
