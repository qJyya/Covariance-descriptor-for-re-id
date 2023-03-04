# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import pdb
import math
import pdb

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--test_dir',default='/media/data4/qiujy/reid/Market-1501',type=str, help='./test_data')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
opt = parser.parse_args()
batchsize=opt.batchsize
test_dir = opt.test_dir
data_dir = test_dir
r=1e-7

h, w = 256, 128

data_transforms = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize,
                                             shuffle=False, num_workers=16) for x in ['gallery','query']}

def index(n,h,w):
    list_x=[]
    list_y=[]

    for j in range(h):
        for k in range(w):
            list_y.append(j)
            list_x.append(k)
    fy=torch.tensor(list_y).reshape(1,len(list_y))
    fx=torch.tensor(list_x).reshape(1,len(list_x))

    ty=fy
    tx=fx
    for i in range(n-1):
        ty=torch.cat((ty,fy),dim=1)   
        tx=torch.cat((tx,fx),dim=1)
    
    return ty.to(device),tx.to(device)

def extract_feature(dataloaders):
    
    cov_list=[]

    for iter,data in enumerate(dataloaders):
        print("进度：%d / %d"%(iter+1,len(dataloaders)))
        imgs, label = data

        #图片为 nx3x256x128 
        n, c, h, w = imgs.size()
        imgs=imgs.double().to(device)
          
        if iter==0 or (iter+1)==len(dataloaders):
             ty,tx=index(n,h,w)
   
        new_imgs=imgs[0].reshape(3,h*w).to(device)

        #重新拼接batch
        for k in range(1,n):
            # 3 x n*h*w
            new_imgs=torch.cat((new_imgs,imgs[k].reshape(3,h*w)),dim=1).double().to(device)    
        
        #imgs0 = imgs.reshape(n, c, h*w)
        #o1 = (imgs0[:,0,:] - imgs0[:,1,:])/math.sqrt(2)
        #o2 = (imgs0[:,0,:] + imgs0[:,1,:] - 2*imgs0[:,2,:])/math.sqrt(6)
        #o3 = (imgs0[:,0,:] + imgs0[:,1,:] + imgs0[:,2,:])/math.sqrt(3)
        
        #1 x n*h*w
        O1=(new_imgs[0]-new_imgs[1])/math.sqrt(2)
        O2=(new_imgs[0]+new_imgs[1]-2*new_imgs[2])/math.sqrt(6)
        O3=(new_imgs[0]+new_imgs[1]+new_imgs[2])/math.sqrt(3)
        
        div1=O1/O3
        div2=O2/O3
        
        matrix=torch.tensor([
                [0.06,0.63,0.27],
                [0.30,0.04,-0.35],
                [0.34,-0.60,0.17]
        ]).double().to(device)
       

        E=torch.mm(matrix,new_imgs).double().to(device)
        E=E.reshape(n*3,h,w).to(device)

        #reshape移动，再拉直 
        tmp_x=torch.zeros(n*3,h,w).to(device)
        tmp_x[:,:,0:w-1]=E[:,:,1:w]
        Ex=(((tmp_x-E).reshape(3,n*h*w))).to(device)
        aEx=abs(Ex).to(device)
          
        tmp_y=torch.zeros(n*3,h,w).to(device)
        tmp_y[:,0:h-1,:]=E[:,1:h,:]
        Ey=((tmp_y-E).reshape(3,n*h*w)).to(device)
        aEy=abs(Ey).to(device)

        arc=torch.atan(Ey/Ex).to(device)


        #拼接各个矩阵
        feature=torch.cat((tx,ty,div1.reshape(1,n*w*h),div2.reshape(1,n*w*h),aEx[0].reshape(1,n*w*h),aEy[0].reshape(1,n*w*h),
        arc[0].reshape(1,n*w*h),aEx[1].reshape(1,n*w*h),aEy[1].reshape(1,n*w*h),arc[1].reshape(1,n*w*h),
        aEx[2].reshape(1,n*w*h),aEy[2].reshape(1,n*w*h),arc[2].reshape(1,n*w*h)),dim=0)

        '''
        for start in range(n):
            end=min((start+1)*opt.batchsize, len(dataloaders.dataset))
            cov_list.append(torch.cov(feature[:,end]))
        '''
        #求各个图片的协方差矩阵
        for s in range(n):
            start=s*h*w
            end=min((s+1)*h*w, len(dataloaders.dataset)*h*w)
            #print(torch.cov(feature[:,start:end]))
            cov_list.append(torch.cov(feature[:,start:end]))
    
    cov_feature=(torch.stack(cov_list,dim=0))
    return cov_feature   
        
def get_id(img_path):   
    camera_id = []
    labels = []
    for path, v in img_path:
        #filename = path.split('/')[-1]
        #会把后缀去掉
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2]=='-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels

gallery_path = image_datasets['gallery'].imgs
query_path = image_datasets['query'].imgs

gallery_cam,gallery_label = get_id(gallery_path)
query_cam,query_label = get_id(query_path)


# Extract feature
since = time.time()

query_feature = extract_feature(dataloaders['query'])
gallery_feature = extract_feature(dataloaders['gallery'])
    
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.2f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
# Save to Matlab for check
result = {'gallery_f':gallery_feature.cpu().numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.cpu().numpy(),'query_label':query_label,'query_cam':query_cam}
scipy.io.savemat('Market_result.mat',result)
