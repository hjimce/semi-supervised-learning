#coding=utf-8
import os
from PIL import Image
import  numpy as np
with open('data/white/val.txt', 'r') as f:
    for line in f.readlines():  # 读取label文件
        path=os.path.join("./", line.split()[0])
        image=Image.open(path)#\
        image=np.asarray(image)
        if len(image.shape)<=2:
            print image.shape
            os.remove(path)
         #   .resize((256, 256))
        #image.save(path)
        #print image.size