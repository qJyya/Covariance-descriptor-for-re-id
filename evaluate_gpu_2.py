import scipy.io
import torch
import numpy as np
#import time
import os 
import math
import pdb
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#求两个协方差矩阵的广义特征值
def eigenvalues(K,M):
    # M进行cholesky分解 M=L*L^T
    # M是 400k x 13 x 13
    # K是 13 x 13
    L = torch.linalg.cholesky(M.to(device), upper=False)

    L.to(device)
    L_inv = torch.linalg.inv(L)
 
    # B=L^-1*K
    B = torch.matmul(L_inv,K)
    
    B.to(device)
    
    A = torch.bmm(L_inv,torch.transpose(B,1,2))
    A.to(device)

    # 此时 K*U = w^2*M*U 转化为 A*y=λ*y   L*x=y
    # λ是一个特征值tensor [420000,13]
    λ = torch.linalg.eigvals(A).real
   
    return λ
    
#######################################################################
# Evaluate
def evaluate(qf,ql,qc,gf,gl,gc):
    # query = qf.to(device)
    # n=gf.shape[0]
    # dist_list=[]
    # for i in range(n):
    #     λ=(eigenvalues(query.to(device),gf[i].to(device))).to(device)
    #     #print(λ)
    #     dist=(torch.sqrt((torch.pow(torch.log(abs(λ)),2)).sum())).item()
    #     dist_list.append(dist)
    # dist=np.array(dist_list)

    λ=(eigenvalues(qf.to(device),gf.to(device))).to(device)
    dist_tensor=torch.sqrt(torch.sum((torch.pow(torch.log(λ),2)),dim=1))
    dist=dist_tensor.cpu().numpy()    
    
    #qf是13x13,gf是40万个13x13
    #λ=eigenvalues(qf,gf)
   
    """ query = qf.view(-1,1)
    # print(query.shape)
    score = torch.mm(gf,query)
    score = score.squeeze(1).cpu()
    score = score.numpy() """
    # predict index
    index = np.argsort(dist)  #from small to large
 
    #index = index[::-1]
    # index = index[0:2000]
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
    cmc = torch.IntTensor(len(index)).zero_().to(device)
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
    since = time.time()
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.2f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

    return ap, cmc

######################################################################
#result = scipy.io.loadmat('Market_result.mat')
#result = scipy.io.loadmat('iLIDS_result.mat')
#result = scipy.io.loadmat('WGU_result.mat')
#result = scipy.io.loadmat('city2m_part1.mat')
#result = scipy.io.loadmat('city2m_part2.mat')
#result = scipy.io.loadmat('city1m_part1_resize.mat')
result = scipy.io.loadmat('city1m_part2_resize_cut.mat')

query_feature = torch.FloatTensor(result['query_f']) #tensor数据
query_cam = result['query_cam'][0] #列表
query_label = result['query_label'][0] #列表,[[]]的形式
gallery_feature = torch.FloatTensor(result['gallery_f'])
gallery_cam = result['gallery_cam'][0]
gallery_label = result['gallery_label'][0]


query_feature = query_feature.to(device)
gallery_feature = gallery_feature.to(device)

print(query_feature.shape)
CMC = torch.IntTensor(len(gallery_label)).zero_().to(device)
ap = 0.0
#print(query_label)
for i in range(len(query_label)):
    ap_tmp, CMC_tmp = evaluate(query_feature[i].to(device),query_label[i],query_cam[i],gallery_feature.to(device),gallery_label,gallery_cam)
    if CMC_tmp[0]==-1:
        continue
    CMC = CMC + CMC_tmp
    ap += ap_tmp
    print(i, CMC_tmp[0])

CMC = CMC.float()
CMC = CMC/len(query_label) #average CMC
print('Rank@1:%f Rank@5:%f Rank@10:%f Rank@20:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],CMC[19],ap/len(query_label)))

