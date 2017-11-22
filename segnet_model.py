import torch.nn as nn
import torch
import os
import numpy as np
import torchvision.utils as vutils
import math

class down_block(nn.Module):
    #using the input channels I specify the channels at for repeated use of this block
    def __init__(self, channels, num_of_convs = 2):
        super(down_block,self).__init__()

        self.num_of_convs = num_of_convs

        if(num_of_convs == 2):
            self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=(3,3),stride=1,padding=1,dilation=1,bias=True)
            self.conv2 = nn.Conv2d(channels[1], channels[1], kernel_size=(3,3),stride=1,padding=1,dilation=1,bias=True)


        elif(num_of_convs == 3):
            self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=(3,3),stride=1,padding=1,dilation=1,bias=True)
            self.conv2 = nn.Conv2d(channels[1], channels[1], kernel_size=(3,3),stride=1,padding=1,dilation=1,bias=True)
            self.conv3 = nn.Conv2d(channels[1], channels[1], kernel_size=(3,3),stride=1,padding=1,dilation=1,bias=True)

        self.batchNorm = nn.BatchNorm2d(channels[1])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2), stride=2, return_indices = True)

        
        # Initialize Kernel weights with a normal distribution of mean = 0 , stdev = sqrt(2. / n)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                
    #forward function through the block
    def forward(self, x):
        if(self.num_of_convs == 2):
            fwd_map = self.conv1(x)
            fwd_map = self.batchnorm(fwd_map)
            self.relu(fwd_map)

            fwd_map = self.conv2(fwd_map)
            fwd_map = self.batchnorm(fwd_map)
            self.relu(fwd_map)

        elif(self.num_of_convs == 3):
            fwd_map = self.conv1(x)
            fwd_map = self.batchnorm(fwd_map)
            self.relu(fwd_map)

            fwd_map = self.conv2(fwd_map)
            fwd_map = self.batchnorm(fwd_map)
            self.relu(fwd_map)

            fwd_map = self.conv3(fwd_map)
            fwd_map = self.batchnorm(fwd_map)
            self.relu(fwd_map)

        #Saving the tensor to map it to the layers deeper in the model
        fwd_map, indices = self.maxpool(fwd_map)
        return (fwd_map, indices)
                
class up_block(nn.Module):

    def __init__(self,channels,num_of_convs = 2):
        super(up_block,self).__init__()
        
        self.num_of_convs = num_of_convs
        
        #Upsampling
        self.unpooled = nn.MaxUnpool2d((2,2) , stride=2)
        self.unconv = nn.Conv2d(channels[0], channels[1], kernel_size=(8,8), stride=0, padding=1, dilation=1, bias=True)
            
        if(num_of_convs== 2):
            self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=(3,3),stride=1,padding=1,dilation=1,bias=True)
            self.conv2 = nn.Conv2d(channels[1], channels[1], kernel_size=(3,3),stride=1,padding=1,dilation=1,bias=True)

        elif(num_of_convs == 3):
            self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=(3,3),stride=1,padding=1,dilation=1,bias=True)
            self.conv2 = nn.Conv2d(channels[1], channels[1], kernel_size=(3,3),stride=1,padding=1,dilation=1,bias=True)
            self.conv3 = nn.Conv2d(channels[1], channels[1], kernel_size=(3,3),stride=1,padding=1,dilation=1,bias=True)
        
        self.batchNorm = nn.BatchNorm2d(channels[1])
        self.relu = nn.ReLU(inplace=True)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

                
    #forward function through the block
    def forward(self, x, indices):
        
        fwd_map = self.unpooled(x, indices)
        fwd_map = self.unconv(fwd_map)
    
        if(self.num_of_convs == 2):
            fwd_map = self.conv1(fwd_map)
            fwd_map = self.batchnorm(fwd_map)
            self.relu(fwd_map)

            fwd_map = self.conv2(fwd_map)
            fwd_map = self.batchnorm(fwd_map)
            self.relu(fwd_map)

        elif(self.num_of_convs == 3):
            fwd_map = self.conv1(fwd_map)
            fwd_map = self.batchnorm(fwd_map)
            self.relu(fwd_map)

            fwd_map = self.conv2(fwd_map)
            fwd_map = self.batchnorm(fwd_map)
            self.relu(fwd_map)

            fwd_map = self.conv3(fwd_map)
            fwd_map = self.batchnorm(fwd_map)
            self.relu(fwd_map)

        return fwd_map
class network(nn.Module):

    def __init__(self):
        super(network,self).__init__()
        self.layer1 = down_block((3,64), 2)
        self.layer2 = down_block((64,128), 2)
        self.layer3 = down_block((128,256), 3)
        self.layer4 = down_block((256,512), 3)
        self.layer5 = down_block((512,512), 3)
        
        self.layer6 = up_block((512,512), 3)
        self.layer7 = up_block((512,256), 3)
        self.layer8 = up_block((256,128), 3)
        self.layer9 = up_block((128,64), 2)
        self.layer10 = up_block((64,10), 2)
        
        self.softmax = nn.Softmax()

    def forward(self,x):
        out1, indices1 = self.layer1(x)
        out2, indices2 = self.layer2(out1)
        out3, indices3 = self.layer3(out2)
        out4, indices4 = self.layer4(out3)
        out5, indices5 = self.layer5(out4)
        
        out6 = self.layer6(out5, indices5)
        out7 = self.layer7(out6, indices6)
        out8 = self.layer8(out7, indices7)
        out9 = self.layer9(out8, indices8)
        out10 = self.layer10(out9, indices9)
        
        res = self.softmax(out10)
        
        return res 
