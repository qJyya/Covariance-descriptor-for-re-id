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

        self.data_transform = transforms.Compose([
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
        
    def get_img(self,base,path):
        imgs_path=[]
        pkl=open(path,'rb')
        p=pickle.load(pkl)

        for i in range(len(p[0])):
            # path,group_label,box
            imgs_path.append((os.path.join(base,'images',p[0][i]),int(p[1][i]),p[3][i]))

        return imgs_path

    # def __getitem__(self,index):
    #     img_path,label,boxs=self.imgs[index]
    #     img=Image.open(img_path)
    #     min_w=self.w
    #     min_h=self.h
    #     max_w=0
    #     max_h=0
        
    #     for box in boxs:
    #         if box[0]<min_w:
    #             min_w=box[0]
    #         if box[2]<min_w:
    #             min_w=box[2]
    #         if box[0]>max_w:
    #             max_w=box[0]
    #         if box[2]>max_w:
    #             max_w=box[2]

    #         if box[1]<min_h:
    #             min_h=box[1]
    #         if box[3]<min_h:
    #             min_h=box[3]
    #         if box[1]>max_h:
    #             max_h=box[1]
    #         if box[3]>max_h:
    #             max_h=box[3]

    #     img=img.crop((min_w,min_h,max_w,max_h))
    #     img=self.data_transform(img)

    #     return img,label


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
