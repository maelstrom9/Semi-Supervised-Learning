
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms, utils, datasets, models
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_ft = models.resnet50(pretrained=False)
num_ftrs = model_ft.fc.in_features
model_ft.fc = torch.nn.Linear(num_ftrs, 200)
model_ft.load_state_dict(torch.load("ssl_best_model.pt",map_location=torch.device('cpu')))
model_ft.to(device)
model_ft.eval()

data_transforms = {
     "val" : transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
}

ids = []
res = []

# path = '/Users/sai/Downloads/semi-inat-2020/test/test/'
path = '../test/test'


for im in tqdm(os.listdir(path)):
  # print(im)
  ids.append(im[:-4])
  image = Image.open(os.path.join(path,im)).convert('RGB')
  x = data_transforms['val'](image)
  x = x.unsqueeze(0)
  pix = np.array(x) #convert image to numpy array
  # print((image))
  # image.show()

  img = torch.from_numpy(pix).to(device)
  outputs = model_ft(img)
  outputs = torch.softmax(outputs,-1)
  # print(torch.argmax(outputs))
  pred = outputs.topk(5,1,largest=True,sorted=True)[1]
  # print(pred)
  pred = pred.cpu().numpy().tolist()
  pred = pred[0]
  pred = [str(i) for i in pred]
  res.append(' '.join(pred))

import pandas as pd
df = pd.DataFrame({"Id":ids,"Category":res})
df.to_csv("sub.csv",index=None)