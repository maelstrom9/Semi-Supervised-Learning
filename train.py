import os
import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, utils, datasets, models
from torch.utils.tensorboard import SummaryWriter
import argparse


parser = argparse.ArgumentParser(description='Load config data')
parser.add_argument('--path', type=str, help='training data path',required=True)
parser.add_argument('--train_size',type= int,help='total no of train image pairs to be sampled per epoch',required=True)
parser.add_argument('--val_size',type = int,help='total no of val image pairs to be sampled per epoch',required=True)
parser.add_argument('--train_bs',type = int,help='train batch size',default=128)
parser.add_argument('--val_bs',type = int,help='val batch size',default=64)
parser.add_argument('--num_workers',type = int,help='no of workers in data loader',default=4)
parser.add_argument('--lr',type=float,help = 'learning rate',default=0.001)
parser.add_argument('--reg',type=float,help='regulirazation',default=1e-6)
parser.add_argument('--epochs',type=int,help='no of epochs',default=100)

args = parser.parse_args()

## setup tensorboard to view train,val losses

writer = SummaryWriter()

from model import Siamese
from resnet_model import ResnetSiamese,Identity
from dataloader import SiameseDataset
from attn_model.residual_attention_network import ResidualAttentionModel_92,ResidualAttentionModel_56

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("GPU available :{}".format(device))

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

## dataset and dataloaders

bs = {'train':args.train_bs,'val':args.val_bs}
lengths = {'train':args.train_size,'val':args.val_size}
probs = {'train':0.5,'val':0.5}
# path = '/Users/sai/Downloads/ssl_trainval_data/'
image_datasets = {x: SiameseDataset(os.path.join(args.path,x),data_transforms[x],lengths[x],probs[x],bs[x]) for x in ['train','val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train','val']}
data_loaders = {'train': torch.utils.data.DataLoader(image_datasets['train'],batch_size=args.train_bs,num_workers=args.num_workers,shuffle=True),
               'val': torch.utils.data.DataLoader(image_datasets['val'],batch_size=args.val_bs,num_workers=args.num_workers,shuffle=True)}

## model

# resn = models.resnet50(pretrained=True)
# num_ftrs = resn.fc.in_features
# resn.fc = nn.Linear(num_ftrs, 200)
# resn.load_state_dict( torch.load("baseline_weights.pt"))
# # resn =  models.resnet50(pretrained=True)
# # model = Siamese(3)
# # resn.fc = Identity()

resn = ResidualAttentionModel_92()
resn.fc = Identity()

model = ResnetSiamese(3,resn)
# model = Siamese(3)
model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.reg)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)


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
            # if phase == 'train':
            #     scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == "train":
                writer.add_scalar('training loss', epoch_loss , epoch)
                writer.add_scalar('training accuracy', epoch_acc, epoch)
            else:
                writer.add_scalar('val loss', epoch_loss, epoch)
                writer.add_scalar('val accuracy', epoch_acc, epoch)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, "best_model.pt")

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "best_model.pt")
    return model


train(model,criterion,optimizer,exp_lr_scheduler,args.epochs)

