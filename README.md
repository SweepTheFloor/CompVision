This is my pytorch implementation of a Segmentation Network model based on the following paper:
https://arxiv.org/abs/1511.00561

This segmentation algorithm has two sections the downsampling part called the ENCODER and the upsampling part called the DECODER
![alt text](Screenshots/Segnet.png "Description goes here")

For the ENCODER part the CNN weights from the classification network vgg16 were extracted and added to tour segnet model

The segmentation model will be trained to recognize roads from backgrounds using a dataset from Kitti:
http://www.cvlibs.net/datasets/kitti/eval_road.php
