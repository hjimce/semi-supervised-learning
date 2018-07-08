#coding=utf-8
import torch
from torch.utils.data import DataLoader, Dataset,sampler
import os
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
import numpy as np
#数据变换
data_transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomRotation(30),
    torchvision.transforms.RandomCrop((32,32)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
data_transform_eval=torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop((32,32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
#对于自定义的数据，要重载下面三个函数getitem、len、init
class mydata(Dataset):
    def __init__(self,label_file,image_root,is_train=True):
        self.imagepaths=[]
        self.labels=[]
        self.is_train=is_train
        if is_train:
            self.transforms=data_transform_train
        else:
            self.transforms=data_transform_eval

        with open(label_file,'r') as f:
            for line in f.readlines():#读取label文件
                self.imagepaths.append(os.path.join(image_root,line.split()[0]))
                self.labels.append(int(line.split()[1]))
    def __getitem__(self, item):

        x=Image.open(self.imagepaths[item]).resize((35,35))

        y=self.labels[item]
        if self.is_train:
            return [self.transforms(x),self.transforms(x)], y
        else:
            return self.transforms(x),y
    def __len__(self):
        return len(self.imagepaths)

def make_weights_for_balanced_classes(labels, nclasses):
    count = {}
    for item in labels:
        if count.has_key(item):
            count[item] += 1
        else:
            count[item]=1
    weight_per_class ={}
    N = len(labels)
    for key,value in count.items():
        weight_per_class[key] = N/float(value)
    weight = [0] * len(labels)
    for idx, val in enumerate(labels):
        weight[idx] = weight_per_class[val]
    return weight


'''train_data=mydata('data/white/val.txt','./',is_train=True)
weights = make_weights_for_balanced_classes(train_data.labels, 3)
weights = torch.DoubleTensor(weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
train_dataloader_student=DataLoader(train_data,batch_size=60,sampler=sampler)
for x,y in train_dataloader_student:
    for xi in x:
        print y
        npimg = torchvision.utils.make_grid(xi).numpy()#可视化显示
        #plt.imshow(np.transpose(npimg, (1, 2, 0)))
        #plt.show()'''

