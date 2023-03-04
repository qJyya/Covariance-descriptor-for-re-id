import scipy.io
import torch
import numpy as np
#import time
import os
import numpy as np
import scipy.linalg as sp_linalg
import math
import pdb

#求两个协方差矩阵的广义特征值
def eigenvalues(K,M):
    # M进行cholesky分解 M=L*L^T
    L = np.linalg.cholesky(M)
    L_inv = np.linalg.inv(L)
    # B=L^-1*K
    B = np.dot(L_inv,K)
    A = np.dot(L_inv,B.T)

    # 此时 K*U = w^2*M*U 转化为 A*y=λ*y   L*x=y
    eig = np.linalg.eig(A)

    # λ是一个特征值列表
    λ = eig[0]
    
    return λ

#######################################################################
# Evaluate
def evaluate(qf,ql,qc,gf,gl,gc):
    query = qf
    n=gf.shape[0]
    dist_list=[]

    for i in range(n):
        λ=eigenvalues(query,gf[i])
        sum=0
        for item in λ:
            if item==0:
                item=1e-9
            sum+=math.pow(math.log(abs(item)),2)
        dist_list.append(math.sqrt(sum))
    dist=np.array(dist_list)
    #score = np.dot(gf,query)

    # predict index
    index = np.argsort(dist)  #from small to large
    #index = index[::-1]
    #index = index[0:2000]
    # good index
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl==-1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1) #.flatten())
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
 
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1

    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc

######################################################################
#result = scipy.io.loadmat('Market_result.mat')
#result = scipy.io.loadmat('iLIDS_result.mat')
#result = scipy.io.loadmat('city2m_part1.mat')
#result = scipy.io.loadmat('city2m_part2.mat')
result = scipy.io.loadmat('city1m_part1_resize.mat')
#result = scipy.io.loadmat('city1m_part2_resize.mat.mat')
query_feature = result['query_f']
query_cam = result['query_cam'][0]
query_label = result['query_label'][0]
gallery_feature = result['gallery_f']
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]

CMC = torch.IntTensor(len(gallery_label)).zero_()
ap = 0.0

for i in range(len(query_label)):

    ap_tmp, CMC_tmp = evaluate(query_feature[i],query_label[i],query_cam[i],gallery_feature,gallery_label,gallery_cam)
    if CMC_tmp[0]==-1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp
    print(i, CMC_tmp[0])

CMC = CMC.float()
CMC = CMC/len(query_label)   #average CMC
print('Rank@1:%f Rank@5:%f Rank@10:%f Rank@20:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],CMC[19],ap/len(query_label)))
