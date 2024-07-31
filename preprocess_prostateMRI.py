import shutil
import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
from torch.nn import functional as F
import pickle
import random
import pandas as pd
from tqdm import tqdm
import cv2
import SimpleITK as sitk
from scipy.ndimage import zoom
import json

base_dir = "data"
dataset_mapping = {"RUNMC": {"site": "A", "test_fd":['0003', '0008', '0012', '0015', '0018', '0026']}, 
                   "BMC": {"site": "B", "test_fd":['0003', '0008', '0012', '0015', '0018', '0026']}, 
                   "I2CVB": {"site": "C", "test_fd":['0003', '0008', '0013', '0015']},
                   "UCL": {"site": "D", "test_fd":['0001', '0032', '0034']},
                   "BIDMC": {"site": "E", "test_fd":['0000', '0004', '0009']},
                   "HK": {"site": "F", "test_fd":['0038', '0041', '0046']}}


def save_df(fd_list, save_pth, df_pth, sort=False):
    path_list_all = []
    for data_fd in fd_list:
        slice_list = os.listdir(save_pth+'/'+data_fd+'/images')
        if sort: slice_list.sort()
        slice_pth_list = [data_fd+'/images/'+slice for slice in slice_list]
        path_list_all = path_list_all + slice_pth_list
    
    if not sort:
        for _ in range(5):
            random.shuffle(path_list_all)

    df = pd.DataFrame(path_list_all, columns=['image_pth'])
    df['mask_pth'] = path_list_all
    df['mask_pth'] = df['mask_pth'].apply(lambda x: x.replace('/images/2Dimage_', '/masks/2Dmask_'))
    df.to_csv(df_pth, index=False)

def get_csv(site, test_fd_list):          
    save_pth = base_dir + '/prostate/2D_all_5slice_site'+site
    
    training_csv = save_pth+'/training.csv'
    validation_csv = save_pth+'/validation.csv'
    test_csv = save_pth+'/test.csv'
    all_csv = save_pth+'/all.csv'

    data_fd_list = os.listdir(save_pth)
    data_fd_list = [data_fd for data_fd in data_fd_list if data_fd.startswith('00') and '.' not in data_fd]

    for _ in range(5):    
        random.shuffle(data_fd_list)
    
    training_fd_list = list(set(data_fd_list)-set(test_fd_list))
    validation_fd_list = random.sample(test_fd_list, min(len(test_fd_list), 4))

    save_df(data_fd_list, save_pth, all_csv)
    save_df(training_fd_list, save_pth, training_csv)
    save_df(validation_fd_list, save_pth, validation_csv)
    save_df(test_fd_list, save_pth, test_csv, sort=True)

def get_all_5slice():
    save_pth = base_dir + '/prostate/2D_all_5slice_site'
    data_pth_all = [os.path.join(base_dir, key) for key in dataset_mapping.keys()]

    for data_pth in data_pth_all:
        site = dataset_mapping[os.path.basename(data_pth)]["site"]
        test_fd_list = dataset_mapping[os.path.basename(data_pth)]["test_fd"]
        data_fd_list = os.listdir(data_pth)
        data_fd_list = [data_fd for data_fd in data_fd_list if data_fd.endswith('tion.nii.gz')]
        data_fd_list.sort()

        cnt = 0
        for data_fd_indx, data_fd in enumerate(data_fd_list):
            case_id = data_fd[4:6]

            if not os.path.exists(save_pth+site+'/00'+case_id):
                os.makedirs(save_pth+site+'/00'+case_id)
                os.mkdir(save_pth+site+'/00'+case_id+'/images')
                os.mkdir(save_pth+site+'/00'+case_id+'/masks')
            
            #load mask
            mask_obj = nib.load(data_pth + '/' + data_fd)
            mask_arr = mask_obj.get_fdata()
            mask_arr[mask_arr>1] = 1
            #load image
            img_obj = nib.load(data_pth + '/' + data_fd.split('_')[0]+'.nii.gz')
            img_arr = img_obj.get_fdata()

            img_arr = np.float32(img_arr)
            mask_arr = np.float32(mask_arr)

            high = np.quantile(img_arr, 0.99)
            low = np.min(img_arr)
            img_arr = np.where(img_arr > high, high, img_arr)
            lungwin = np.array([low * 1., high * 1.])
            img_arr = (img_arr - lungwin[0]) / (lungwin[1] - lungwin[0])  
        
            # h, w  = img_arr.shape[0], img_arr.shape[1]
            # out_h, out_w = 512, 512

            # if h != 512 or w !=512:
            #     print(h, w)
            #     img_arr = zoom(img_arr, (out_h / h, out_w / w, 1.0), order=3)
            #     mask_arr = zoom(mask_arr, (out_h / h, out_w / w, 1.0), order=0)

            img_arr = np.concatenate((img_arr[:, :, 0:1], img_arr[:, :, 0:1], img_arr, img_arr[:, :, -1:], img_arr[:, :, -1:]), axis=-1)
            mask_arr = np.concatenate((mask_arr[:, :, 0:1], mask_arr[:, :, 0:1], mask_arr, mask_arr[:, :, -1:], mask_arr[:, :, -1:]), axis=-1)    

            for slice_indx in range(2, img_arr.shape[2]-2):
                
                slice_arr = img_arr[:,:,slice_indx-2: slice_indx+3]
                slice_arr = np.flip(np.rot90(slice_arr, k=1, axes=(0, 1)), axis=1)

                mask_arr_2D = mask_arr[:,:,slice_indx-2: slice_indx+3]
                mask_arr_2D = np.flip(np.rot90(mask_arr_2D, k=1, axes=(0, 1)), axis=1)

                with open(save_pth+site+'/00'+case_id+'/images'+'/2Dimage_'+'{:04d}'.format(slice_indx-2)+'.pkl', 'wb') as file:
                    pickle.dump(slice_arr, file)

                with open(save_pth+site+'/00'+case_id+'/masks'+'/2Dmask_'+'{:04d}'.format(slice_indx-2)+'.pkl', 'wb') as file:
                    pickle.dump(mask_arr_2D, file)

            cnt += 1
        get_csv(site, test_fd_list)

if __name__=="__main__":
    get_all_5slice()
