from torch.utils.data import dataset
import pdb
from torchvision import transforms
from PIL import Image
import os
import torch.nn.functional as F
from shutil import copyfile
import pickle
import torch

class LoadData(dataset.Dataset):

    #载入需要传入的文档，对数据进行预处理
    def __init__(self,base,path,h,w):
        super(LoadData,self).__init__()
        self.imgs=self.get_img(base,path)
        self.h=h
        self.w=w
        #预处理
        self.tensor_transform=transforms.ToTensor()
        
    def get_img(self,base,path):
        imgs_path=[]
        with open(path,'r') as f:
            imgs_info=f.readlines()
            for line in imgs_info:
                l=line.strip().split(' ')
                boxs=l[2:]
                new_boxs=[]
                for box in boxs:
                    new_boxs.append(box.split(','))
                imgs_path.append((os.path.join(base,l[0]),l[1],new_boxs))
        self.imgs=imgs_path
        return imgs_path

    def __getitem__(self,index):
        #box给的是左上角和右下角的坐标, lw,lh,rw,rh
        img_path,label,boxs=self.imgs[index]
        img=Image.open(img_path)
        w,h=img.size  # w * h
      
        img=self.tensor_transform(img)
        mask=torch.zeros_like(img)

        #一个box中有多个人的框
        for box in boxs: 
          mask[:,int(box[1]):int(box[3])+1,int(box[0]):int(box[2])+1]=1

        #乘以mask，只取固定区域
        img=torch.mul(mask,img)

        #补成相应的大小
        std_h,std_w=self.h,self.w
        y=std_w-w
        x=std_h-h
        img=F.pad(img,(0,y,0,x),'constant',0)
        
        label=int(label)
        
        return img,label
    
    def __len__(self):
        return len(self.imgs)

r=1e-9
mmmm=torch.ones(3,3,4)
nnnn=torch.randn(mmmm.size())*r
