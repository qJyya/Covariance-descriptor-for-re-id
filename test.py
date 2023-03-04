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

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--gpu_ids',default='7', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch',default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir',default='/media/data4/qiujy/reid/Market-1501',type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
parser.add_argument('--linear_num', default=512, type=int, help='feature dimension: 512 or default or 0 (linear=False)')
parser.add_argument('--use_dense', action='store_true', help='use densenet121' )
parser.add_argument('--use_efficient', action='store_true', help='use efficient-b4' )
parser.add_argument('--use_hr', action='store_true', help='use hr18 net' )
parser.add_argument('--PCB', action='store_true', help='use PCB' )
parser.add_argument('--multi', action='store_true', help='use multiple query' )
parser.add_argument('--fp16', action='store_true', help='use fp16.' )
parser.add_argument('--ibn', action='store_true', help='use ibn.' )
parser.add_argument('--ms',default='1', type=str,help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')

opt = parser.parse_args()
str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir


gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >=0:
        gpu_ids.append(id)

opt = parser.parse_args()
r=1e-9
name = opt.name
test_dir = opt.test_dir

h, w = 256, 128

data_transforms = transforms.Compose([
        transforms.Resize((h, w), interpolation=3),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

data_dir = test_dir

image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir,x) ,data_transforms) for x in ['gallery','query']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=False, num_workers=16) for x in ['gallery','query']}

def extract_feature(dataloaders):
    imgs_feature_list=[]
    for iter,data in enumerate(dataloaders):
        print("进度：%d / %d"%(iter+1,len(dataloaders)))
        imgs, label = data
        imgs=imgs.double()

        #单张图片为 3x256x128 
        count=0
        n, c, h, w = imgs.size() 
        img_feature=torch.DoubleTensor(h*w,13).zero_()

        #为计算梯度用三维数组存储Ei
        E_array=np.zeros((h,w,3))
        
        for k in range(n):
            img=imgs[k,:,:,:]
            for i in range(h):
                for j in range(w):
                    R_feature=img[0,:,:]
                    G_feature=img[1,:,:]
                    B_feature=img[2,:,:]
            
                    R=R_feature[i][j].item()
                    G=G_feature[i][j].item()
                    B=B_feature[i][j].item()
            
                    matrix=torch.tensor([
                        [0.06,0.63,0.27],
                        [0.30,0.04,-0.35],
                        [0.34,-0.60,0.17]
                    ])

                    rgb_matrix=torch.tensor([
                        [R],
                        [G],
                        [B]
                    ]
                    )

                    E_matrix=torch.mm(matrix,rgb_matrix)
    
                    E_array[i][j][0]=E_matrix[0][0].item()
                    E_array[i][j][1]=E_matrix[1][0].item()
                    E_array[i][j][2]=E_matrix[2][0].item()
            
            for i in range(1,h-1):
                for j in range(1,w-1):
                    pixel_feature=torch.DoubleTensor(1,13)
                    R_feature=img[0,:,:]
                    G_feature=img[1,:,:]
                    B_feature=img[2,:,:]

                    R=R_feature[i][j].item()
                    G=G_feature[i][j].item()
                    B=B_feature[i][j].item()
                    
                    O_1=(R-G)/math.sqrt(2)
                    O_2=(R+G-2*B)/math.sqrt(6)
                    O_3=(R+G+B)/math.sqrt(3)+r
            
                    div_1=O_1/O_3
                    div_2=O_2/O_3

                    E_x1=abs(E_array[i+1][j][0]-E_array[i][j][0])+r
                    E_x2=abs(E_array[i+1][j][1]-E_array[i][j][1])+r
                    E_x3=abs(E_array[i+1][j][2]-E_array[i][j][2])+r


                    E_y1=abs(E_array[i][j+1][0]-E_array[i][j][0])
                    E_y2=abs(E_array[i][j+1][1]-E_array[i][j][1])
                    E_y3=abs(E_array[i][j+1][2]-E_array[i][j][2])
                    
                    arctan_1=math.atan(E_y1/E_x1)
                    arctan_2=math.atan(E_y2/E_x2)
                    arctan_3=math.atan(E_y3/E_x3)
                    
                    pixel_feature.index_fill_(1, torch.tensor(0), i+r)
                    pixel_feature.index_fill_(1, torch.tensor(1), j+r)
                    pixel_feature.index_fill_(1, torch.tensor(2), div_1+r)
                    pixel_feature.index_fill_(1, torch.tensor(3), div_2+r)
                    pixel_feature.index_fill_(1, torch.tensor(4), E_x1)
                    pixel_feature.index_fill_(1, torch.tensor(5), E_y1+r)
                    pixel_feature.index_fill_(1, torch.tensor(6), arctan_1+r)
                    pixel_feature.index_fill_(1, torch.tensor(7), E_x2)
                    pixel_feature.index_fill_(1, torch.tensor(8), E_y2+r)
                    pixel_feature.index_fill_(1, torch.tensor(9), arctan_2+r)
                    pixel_feature.index_fill_(1, torch.tensor(10), E_x3)
                    pixel_feature.index_fill_(1, torch.tensor(11), E_y3+r)
                    pixel_feature.index_fill_(1, torch.tensor(12),arctan_3+r)
                    img_feature[i:i+1,:]=pixel_feature

            #计算协方差矩阵 
            cov_feature=torch.cov(img_feature.T)
            print(cov_feature)
            print(cov_feature.size())
            #img_feature=img_feature.numpy()
            #cov_feature=np.cov(img_feature.T)
            #cov_feature=torch.from_numpy(cov_feature)
        
            imgs_feature_list.append(cov_feature)    
    # n x h*w x d ,将每一张图片的特征concat  
    imgs_feature=torch.stack(imgs_feature_list,dim=0)

    return imgs_feature
                                

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
result = {'gallery_f':gallery_feature.numpy(),'gallery_label':gallery_label,'gallery_cam':gallery_cam,'query_f':query_feature.numpy(),'query_label':query_label,'query_cam':query_cam}
scipy.io.savemat('pytorch_result.mat',result)

result = './model/%s/result.txt'%opt.name
os.system('python evaluate.py | tee -a %s'%result)
