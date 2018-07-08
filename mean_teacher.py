#coding=utf-8
import torch
from torch import  autograd,nn
from torch.utils.data import DataLoader, Dataset
from data_layer import mydata,make_weights_for_balanced_classes
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as function
import os
import time


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2),
            conv_dw( 32,  64, 1),
            conv_dw( 64, 96, 2),
            conv_dw(96, 96, 1),
            conv_dw(96, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 1),
            nn.AvgPool2d(2),
        )
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        x = self.model(x)
        #print x.shape
        x = x.view(-1, 512)
        x = self.fc(x)
        return x
def update_ema_variables(model, ema_model,alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
def softmax_mse_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_softmax = function.softmax(input_logits, dim=1)
    target_softmax = function.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return function.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes

torch.backends.cudnn.enabled = False
torch.manual_seed(7)

net_student=MobileNet().cuda()
net_teacher=MobileNet().cuda()
for param in net_teacher.parameters():
                param.detach_()
if os.path.isfile('white.pt'):#base精度52\53\54,平均值53
    net_student.load_state_dict(torch.load('white.pt'))





train_data=mydata('data/white/train.txt','./',is_train=True)
min_batch_size=32
weights = make_weights_for_balanced_classes(train_data.labels, 3)
weights = torch.DoubleTensor(weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
train_dataloader=DataLoader(train_data,batch_size=min_batch_size,shuffle=True,num_workers=8)
print train_dataloader

valid_data=mydata('data/white/val.txt','./',is_train=False)
valid_dataloader=DataLoader(valid_data,batch_size=min_batch_size,shuffle=True,num_workers=8)




classify_loss_function = torch.nn.CrossEntropyLoss(size_average=False,ignore_index=-1).cuda()
optimizer = torch.optim.SGD(net_student.parameters(),lr = 0.001, momentum=0.9)




globals_step=0
for epoch in range(10000):
    globals_classify_loss=0
    globals_consistency_loss = 0
    net_student.train()
    start=time.time()
    end=0

    for index,(x,y) in enumerate(train_dataloader):


        optimizer.zero_grad()  #

        x_student=autograd.Variable(x[0]).cuda()
        y=autograd.Variable(y).cuda()
        predict_student=net_student(x_student)

        classify_loss=classify_loss_function(predict_student,y)/min_batch_size
        sum_loss = classify_loss

        x_teacher= autograd.Variable(x[1],volatile=True).cuda()
        predict_teacher = net_teacher(x_teacher)
        ema_logit = autograd.Variable(predict_teacher.detach().data, requires_grad=False)
        consistency_loss =softmax_mse_loss(predict_student,ema_logit)/min_batch_size
        consistency_weight=1
        sum_loss+=consistency_weight*consistency_loss
        globals_consistency_loss += consistency_loss.data[0]

        sum_loss.backward()
        optimizer.step()
        alpha = min(1 - 1 / (globals_step + 1), 0.99)
        update_ema_variables(net_student, net_teacher, alpha)



        globals_classify_loss +=classify_loss.data[0]
        globals_step += 1

    if epoch%5!=0:
        continue

    net_student.eval()
    correct = 0
    total = 0
    for images, labels in valid_dataloader:
        valid_input=autograd.Variable(images,volatile=True).cuda()
        outputs = net_student(valid_input)
        #print outputs.shape
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()
    print "epoch:%d"%epoch,"time:%d"%(time.time()-start),'accuracy %d' % (
        100 * correct / total),"consistency loss:%f"%globals_consistency_loss,'classify loss%f:'%globals_classify_loss
    torch.save(net_student.state_dict(),'white.pt')



