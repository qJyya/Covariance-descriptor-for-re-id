from torch.utils.data import dataset
import pdb
from torchvision import transforms
from PIL import Image
import os
import torch.nn.functional as F
from shutil import copyfile
import pickle
import torch
import numpy as np
import random

class LoadData(dataset.Dataset):

    #载入需要传入的文档，对数据进行预处理
    def __init__(self,base,path,h,w):
        super(LoadData,self).__init__()
        self.imgs=self.get_img(base,path)
        self.h=h
        self.w=w
        #预处理
        #self.tensor_transform=transforms.ToTensor()
        self.data_transform = transforms.Compose([
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
        
    def get_img(self,base,path):
        name=os.path.basename(path)
        final=[]
        imgs_path=[]
        sample=[]
        with open(path,'r') as f:
            imgs_info=f.readlines()
            for line in imgs_info:
                l=line.strip().split(' ')
                path=l[0]
                path_list=l[0].split('_')
                
                dir_str=path_list[1]
                dir=''

                if(dir_str=='pCount2'):
                    dir='Images_2'
                if(dir_str=='pCount3'):
                    dir='Images_3'
                if(dir_str=='pCount4'):
                    dir='Images_4'
                if(dir_str=='pCount5'):
                    dir='Images_5'
                if(dir_str=='pCount6'):
                    dir='Images_6'

                label=int(path_list[0])
                
                boxs=l[1:]
                new_boxs=[]
                for box in boxs:
                    new_boxs.append(box.split(',')[1:5])
                imgs_path.append((os.path.join(base,dir,l[0]),label,new_boxs))
        s = random.sample(range(0, 40000), 2000)
        for item in s:
            sample.append(imgs_path[item])

        if name=='query_part1.txt' or name=='query_part2.txt':
            final=sample
        if name=='gallery_part1.txt' or name=='gallery_part2.txt':
            final=imgs_path
         
        self.imgs=final
        # self.imgs=imgs_path
        return final

    def __getitem__(self,index):
        #box给的是左上角和右下角的坐标, lw,lh,rw,rh
        img_path,label,boxs=self.imgs[index]
        img=Image.open(img_path)
        
        array=np.array(img)
        mask=np.zeros(array.shape)

        for box in boxs: 
            mask[int(float(box[1])):int(float(box[3]))+int(float(box[1]))+1:,
                int(float(box[0])):int(float(box[2]))+int(float(box[0]))+1,:
            ]=1
        img=mask*array

        new_pic=Image.fromarray(np.uint8(img))
        img=self.data_transform(new_pic)
        
        label=int(label)
        
        return img,label
    
    def __len__(self):
        return len(self.imgs)

