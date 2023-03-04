import os
from shutil import copyfile
import pickle

pkl_query=open('/media/data4/qiujy/reid/iLIDS/GReID_label/ilids_query.pkl','rb')
pkl_gallery=open('/media/data4/qiujy/reid/iLIDS/GReID_label/ilids_gallery.pkl','rb')
pkl1=pickle.load(pkl_query)
pkl2=pickle.load(pkl_gallery)

# You only need to change this line to your dataset download path
download_path = '/media/data4/qiujy/reid/iLIDS' 

if not os.path.isdir(download_path):
    print('please change the download_path')

if not os.path.isdir(download_path):
    os.mkdir(download_path)

#-----------------------------------------
#query
query_path = download_path + '/images'
query_save_path = download_path + '/query'
if not os.path.isdir(query_save_path):
    os.mkdir(query_save_path)

for name in pkl1[0]:
    #是否为jpg文件
    print(name)
    if not name[-3:]=='jpg':
        continue
    ID  = name.split('_')
    src_path = query_path + '/' + name
    dst_path = query_save_path + '/' + ID[1] 
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)
    copyfile(src_path, dst_path + '/' + name)

#gallery
gallery_path = download_path + '/images'
gallery_save_path = download_path + '/gallery'
if not os.path.isdir(gallery_save_path):
    os.mkdir(gallery_save_path)

for name in pkl2[0]:
    #是否为jpg文件
    print(name)
    if not name[-3:]=='jpg':
            continue
    ID  = name.split('_')
    src_path = gallery_path + '/' + name
    dst_path = gallery_save_path + '/' + ID[1]
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)
    copyfile(src_path, dst_path + '/' + name)

