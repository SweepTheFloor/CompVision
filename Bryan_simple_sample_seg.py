from segnet_model import network
from data_loader import data_loader_seg

import torch 
import numpy as np 
from torch.autograd import Variable
import torch.nn as nn
from torchvision import datasets,models,transforms
import torch.optim as optim


model_ft = network()

if torch.cuda.is_available():
    model_ft = model_ft.cuda()

#APPLY TRANSFORM IF NEEDED
trans = transforms.Compose([transforms.ToTensor()])

dsets = data_loader_seg('images/training/',trans=trans)
dsets_enqueuer = torch.utils.data.DataLoader(dsets, batch_size=1, num_workers=0, drop_last=False)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model_ft.parameters(),lr = 0.001, betas=(0.9, 0.999), eps=1e-08)

if torch.cuda.is_available():
    criterion = criterion.cuda()

loss_data = 0.0

for idx,data in enumerate(dsets_enqueuer,1):
    image,image_seg = data['image'], data['image_seg']

if torch.cuda.is_available():
    image, image_seg = Variable(image.cuda(), requires_grad = False), Variable(image_seg.cuda(), requires_grad = False)
else:
    image, image_seg = Variable(image, requires_grad = False), Variable(image_seg, requires_grad = False)

output = model_ft(image)
