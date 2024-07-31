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

nnUNet_stats = {
    "A" :{"percentile_00_5": -66.0, "percentile_99_5": 208.0, "mean": 100.54512023925781, "std": 61.67744445800781},
    "B" :{"percentile_00_5": -77.0, "percentile_99_5": 341.0, "mean": 110.72207641601562, "std": 84.01102447509766},
    "C" :{"percentile_00_5": -89.0, "percentile_99_5": 325.0, "mean": 123.67296600341797, "std": 97.30636596679688},
    "D" :{"percentile_00_5": -64.0, "percentile_99_5": 202.0, "mean": 71.06884002685547, "std": 52.98845672607422},
    "E" :{"percentile_00_5": -64.0, "percentile_99_5": 236.0, "mean": 91.36735534667969, "std": 60.98041915893555},
    "F" :{"percentile_00_5": -82.0, "percentile_99_5": 288.0, "mean": 105.31212615966797, "std": 76.49889373779297}
}

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
    save_pth = base_dir + '/kits/2D_all_5slice_site'+site
    
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

def get_all_5slice_npz():
    df = pd.read_csv(base_dir + '/kits19_data/thresholded_sites.csv')
    data_pth = base_dir + '/kits19_data/KiTS2019_site'
    save_pth = base_dir + '/kits/2D_all_5slice_site'
    for i in df.site_ids.unique():
        site = chr(65 + i)
        test_fd_list = df[(df.site_ids == i) & (df.train_test_split == 'test')].case_ids.str[-5:].to_list()
        data_fd_list = df[(df.site_ids == i)].case_ids.to_list()

        for data_fd_indx, data_fd in tqdm(enumerate(data_fd_list), total=len(data_fd_list)):
            case_id = data_fd[-5:]

            if not os.path.exists(os.path.join(save_pth+site, case_id)):
                os.makedirs(os.path.join(save_pth+site, case_id))
                os.mkdir(os.path.join(save_pth+site, case_id,'images'))
                os.mkdir(os.path.join(save_pth+site, case_id,'masks'))
            
            # breakpoint()
            #load data
            data = np.load(f'{data_pth}{site}/{data_fd}.npz')
            #load mask
            mask_arr = data['seg'] 
            #load image
            img_arr = data['data']

            img_arr = np.float32(img_arr[0].transpose(1,2,0))
            mask_arr = np.float32(mask_arr[0].transpose(1,2,0))

            img_arr = (img_arr - np.min(img_arr)) / (np.max(img_arr) - np.min(img_arr))

            img_arr = np.concatenate((img_arr[:, :, 0:1], img_arr[:, :, 0:1], img_arr, img_arr[:, :, -1:], img_arr[:, :, -1:]), axis=-1)
            mask_arr = np.concatenate((mask_arr[:, :, 0:1], mask_arr[:, :, 0:1], mask_arr, mask_arr[:, :, -1:], mask_arr[:, :, -1:]), axis=-1)    

            for slice_indx in range(2, img_arr.shape[2]-2):        
                slice_arr = img_arr[:,:,slice_indx-2: slice_indx+3]
                slice_arr = np.flip(np.rot90(slice_arr, k=1, axes=(0, 1)), axis=1)
                mask_arr_2D = mask_arr[:,:,slice_indx-2: slice_indx+3]
                mask_arr_2D = np.flip(np.rot90(mask_arr_2D, k=1, axes=(0, 1)), axis=1)
                with open(save_pth+site+'/'+case_id+'/images'+'/2Dimage_'+'{:04d}'.format(slice_indx-2)+'.pkl', 'wb') as file:
                    pickle.dump(slice_arr, file)
                with open(save_pth+site+'/'+case_id+'/masks'+'/2Dmask_'+'{:04d}'.format(slice_indx-2)+'.pkl', 'wb') as file:
                    pickle.dump(mask_arr_2D, file)

        get_csv(site, test_fd_list)

def get_all_5slice_raw():
    df = pd.read_csv(base_dir + '/kits19_data/thresholded_sites.csv')
    data_pth = base_dir + '/kits19_raw/KiTS2019_site'
    save_pth = base_dir + '/kits/2D_all_5slice_site'
    for i in df.site_ids.unique():
        site = chr(65 + i)
        test_fd_list = df[(df.site_ids == i) & (df.train_test_split == 'test')].case_ids.str[-5:].to_list()
        data_fd_list = df[(df.site_ids == i)].case_ids.to_list()

        for data_fd_indx, data_fd in tqdm(enumerate(data_fd_list), total=len(data_fd_list)):
            case_id = data_fd[-5:]

            if not os.path.exists(os.path.join(save_pth+site, case_id)):
                os.makedirs(os.path.join(save_pth+site, case_id))
                os.mkdir(os.path.join(save_pth+site, case_id,'images'))
                os.mkdir(os.path.join(save_pth+site, case_id,'masks'))
            
            img_arr = nib.load(f'{data_pth}{site}/imagesTr/{data_fd}_0000.nii.gz').get_fdata()
            mask_arr = nib.load(f'{data_pth}{site}/labelsTr/{data_fd}.nii.gz').get_fdata()

            mask_arr = np.float32(np.clip(mask_arr, 0, 2)).transpose(1,2,0)
            img_arr = np.clip(img_arr, 
                              nnUNet_stats[site]["percentile_00_5"], 
                              nnUNet_stats[site]["percentile_99_5"])
            img_arr = np.float32((img_arr - nnUNet_stats[site]["mean"]) / nnUNet_stats[site]["std"]).transpose(1,2,0)

            img_arr = np.concatenate((img_arr[:, :, 0:1], img_arr[:, :, 0:1], img_arr, img_arr[:, :, -1:], img_arr[:, :, -1:]), axis=-1)
            mask_arr = np.concatenate((mask_arr[:, :, 0:1], mask_arr[:, :, 0:1], mask_arr, mask_arr[:, :, -1:], mask_arr[:, :, -1:]), axis=-1)

            for slice_indx in range(2, img_arr.shape[2]-2):
                slice_arr = img_arr[:,:,slice_indx-2: slice_indx+3]
                slice_arr = np.flip(np.rot90(slice_arr, k=1, axes=(0, 1)), axis=1)
                mask_arr_2D = mask_arr[:,:,slice_indx-2: slice_indx+3]
                mask_arr_2D = np.flip(np.rot90(mask_arr_2D, k=1, axes=(0, 1)), axis=1)
                with open(save_pth+site+'/'+case_id+'/images'+'/2Dimage_'+'{:04d}'.format(slice_indx-2)+'.pkl', 'wb') as file:
                    pickle.dump(slice_arr, file)
                with open(save_pth+site+'/'+case_id+'/masks'+'/2Dmask_'+'{:04d}'.format(slice_indx-2)+'.pkl', 'wb') as file:
                    pickle.dump(mask_arr_2D, file)

        get_csv(site, test_fd_list)

def global_csv():
    path = '2D_all_5slice_site'
    folder_path = base_dir + '/kits/' + path
    df_all_global = pd.DataFrame()
    df_train_global = pd.DataFrame()
    df_val_global = pd.DataFrame()
    df_test_global = pd.DataFrame()

    for site in ['A', 'B', 'C', 'D', 'E', 'F']:
        df_all = pd.read_csv(folder_path+site+'/all.csv')
        df_train = pd.read_csv(folder_path+site+'/training.csv')
        df_val = pd.read_csv(folder_path+site+'/validation.csv')
        df_test = pd.read_csv(folder_path+site+'/test.csv')

        df_all['image_pth'] = df_all['image_pth'].apply(lambda x: path+site+'/'+x)
        df_all['mask_pth'] = df_all['mask_pth'].apply(lambda x: path+site+'/'+x)

        df_train['image_pth'] = df_train['image_pth'].apply(lambda x: path+site+'/'+x)
        df_train['mask_pth'] = df_train['mask_pth'].apply(lambda x: path+site+'/'+x)

        df_val['image_pth'] = df_val['image_pth'].apply(lambda x: path+site+'/'+x)
        df_val['mask_pth'] = df_val['mask_pth'].apply(lambda x: path+site+'/'+x)

        df_test['image_pth'] = df_test['image_pth'].apply(lambda x: path+site+'/'+x)
        df_test['mask_pth'] = df_test['mask_pth'].apply(lambda x: path+site+'/'+x)

        df_all_global = pd.concat([df_all_global, df_all], ignore_index=True)
        df_train_global = pd.concat([df_train_global, df_train], ignore_index=True)
        df_val_global = pd.concat([df_val_global, df_val], ignore_index=True)
        df_test_global = pd.concat([df_test_global, df_test], ignore_index=True)

    df_all_global.sample(frac=1).to_csv(base_dir+'/kits/all.csv', index=False)
    df_train_global.sample(frac=1).to_csv(base_dir+'/kits/training.csv', index=False)
    df_val_global.sample(frac=1).to_csv(base_dir+'/kits/validation.csv', index=False)
    df_test_global.sample(frac=1).to_csv(base_dir+'/kits/test.csv', index=False)

if __name__=="__main__":
    # get_all_5slice_npz()
    get_all_5slice_raw()
    global_csv()
