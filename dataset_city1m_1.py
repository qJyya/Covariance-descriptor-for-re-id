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
          mask[:,int(float(box[1])):int(float(box[3]))+int(float(box[1]))+1,
               int(float(box[0])):int(float(box[2]))+int(float(box[0]))+1]=1

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

""" pic=Image.open('/media/data3/zhangquan/dataset/City1M/Images_5/0001764_pCount5_cam7_0053902_night.jpg')
1116.3201,113.2704,64.8658,160.662

pic1=pic.crop((int(1116.3201),int(113.2704),int(1116.3201+64.8658),int(113.2704+160.662)))

pic1.save('/media/data4/qiujy/reid/reid-cov/b.jpg')
 """

""" path='/media/data3/zhangquan/dataset/City1M/Images_5'
div='imgage1'
mmm='fjeowfij.jpg'
print(os.path.join(path,div,mmm)) """


""" list='1,2,3,4,5,6'
str1=list.split(',')
str2=str1[1:5]
print(str1)
print(str2) """
