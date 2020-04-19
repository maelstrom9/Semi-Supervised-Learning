import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, utils, datasets

from model import Siamese
from dataloader import SiameseDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## datatranforms
data_transforms = {
    "train" : transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(280),
        transforms.RandomResizedCrop(224,(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
     "val" : transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
}

## dataset and dataloaders

path = '/Users/sai/Downloads/ssl_trainval_data/'
image_datasets = {x: SiameseDataset(os.path.join(path,x),data_transforms[x]) for x in ['train','val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train','val']}
data_loaders = {'train': torch.utils.data.DataLoader(image_datasets['train'],batch_size=4,num_workers=1,shuffle=True),
               'val': torch.utils.data.DataLoader(image_datasets['val'],batch_size=4,num_workers=1,shuffle=True)}

## model

model = Siamese(3)
model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01,weight_decay=1e-6)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

epochs = 10


def train(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for sample_batch in data_loaders[phase]:
                images1 = sample_batch["Image1"].to(device)
                images2 = sample_batch["Image2"].to(device)
                targets = sample_batch["target"].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images1,images2)
                    targets = targets.unsqueeze(1).float()
                    loss = criterion(outputs, targets)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * images1.size(0)

                pred_probs = torch.sigmoid(outputs)
                preds = pred_probs>0.5
                running_corrects += torch.sum(preds == targets.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


train(model,criterion,optimizer,exp_lr_scheduler,100)

