import os
import sys
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import numpy as np
import torch
from datasets.dataset import dataset_reader
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from importlib import import_module
from segment_anything import sam_model_registry

from icecream import ic
import pandas as pd
import pickle
from datetime import datetime
from einops import repeat
from scipy.ndimage import zoom
from utils import calculate_metric_percase
import nibabel as nib
from torchinfo import summary
from PIL import Image
# HU_min, HU_max = -200, 250
# data_mean = 50.21997497685108
# data_std = 68.47153712416372

def test_single_volume(image, label, net, classes, multimask_output, patch_size=[512, 512], test_save_path=None, case=None):
    
    image, label = image.squeeze(0), label.squeeze(0) #[b, h, w, d], [b, h, w, d]
    label = label[:,:,:,2]
    
    probability = np.expand_dims(np.zeros_like(label, dtype=np.float32), axis=-1) #[b, h, w, c]
    probability = repeat(probability, 'd h w c -> d h w (repeat c)', repeat=classes+1)

    probability = np.concatenate((probability[0:1], probability[0:1], probability, probability[-1:], probability[-1:]), axis=0)

    avg_cnt = np.ones_like(probability, dtype=np.float32)
    for ind in tqdm(range(image.shape[0])):
        slice = image[ind]
        x, y = slice.shape[0], slice.shape[1]
        if x != patch_size[0] or y != patch_size[1]:
            slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y, 1.0), order=3)
        inputs = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        inputs = repeat(inputs, 'b c h w d -> b (repeat c) h w d', repeat=3)
        inputs = torch.permute(inputs, (0, -1, 1, 2, 3))
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, multimask_output, patch_size[0])
            output_masks = outputs['masks']
            
            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1)
            out = out.cpu().detach().numpy()
            out_pred = torch.softmax(output_masks, dim=1)
            out_pred = torch.permute(out_pred, (0, 2, 3, 1))
            out_pred = out_pred.cpu().detach().numpy()
            out_h, out_w = out.shape[1], out.shape[2]
            if x != out_h or y != out_w:
                out_pred = zoom(out_pred, (1.0, x / out_h, y / out_w, 1.0), order=3)
            
            probability[ind: ind+5] += out_pred
            avg_cnt[ind: ind+5] += 1.
            
    probability = probability/avg_cnt
    prediction = np.argmax(probability, axis=-1)
    prediction = prediction[2:-2]

    metric_list = []
    for i in range(1, classes + 1):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        
        image_data = np.moveaxis(image[:,:,:,2].astype(np.float32), 0, -1)
        prediction_data = np.moveaxis(prediction.astype(np.float32), 0, -1)
        label_data = np.moveaxis(label.astype(np.float32), 0, -1)

        image_data = np.rot90(np.flip(image_data, axis=1), k=-1, axes=(0, 1))
        prediction_data = np.rot90(np.flip(prediction_data, axis=1), k=-1, axes=(0, 1))
        label_data = np.rot90(np.flip(label_data, axis=1), k=-1, axes=(0, 1))

        # Create Nifti images
        img_nifti = nib.Nifti1Image(image_data, np.eye(4))
        prd_nifti = nib.Nifti1Image(prediction_data, np.eye(4))
        lab_nifti = nib.Nifti1Image(label_data, np.eye(4))

        # Set spacing
        img_nifti.header['pixdim'][1:4] = [1, 1, 1]
        prd_nifti.header['pixdim'][1:4] = [1, 1, 1]
        lab_nifti.header['pixdim'][1:4] = [1, 1, 1]

        # Save the images
        img_nifti.to_filename(f"{test_save_path}/{case}_img.nii.gz")
        prd_nifti.to_filename(f"{test_save_path}/{case}_pred.nii.gz")
        lab_nifti.to_filename(f"{test_save_path}/{case}_gt.nii.gz")
        
    return metric_list

def inference_2D(args, multimask_output, model, test_save_path=None):
    color_pallete = {1: [255, 0, 0], 2: [0, 255, 0], 3: [0, 0, 255], 4: [255, 255, 0], 5: [0, 255, 255], 6: [255, 0, 255]}

    db_test = dataset_reader(base_dir=args.data_path, split='test', num_classes=args.num_classes)
    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    model.cuda()
    model.eval()
    metric_list = []
    class_to_name = {1: 'skin lesion', 2: 'globules', 3: 'milia_like_cyst', 4:'negative_network', 5:'pigment_network', 6:'streaks'}
    for sampled_batch in tqdm(test_loader):
        image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        image_batch = image_batch.unsqueeze(1)
        hw_size = image_batch.shape[-1]
        label_batch = label_batch.contiguous().view(-1, hw_size, hw_size)
        image_batch = image_batch.cuda()
        # breakpoint()
        case_name = sampled_batch['case_name'][0].split('/')[-1][:-4]
        with torch.no_grad():
            output = model(image_batch, multimask_output, args.img_size)

        prediction = torch.argmax(output['masks'], dim=1)
        label_batch = label_batch.numpy()
        prediction = prediction.cpu().detach().numpy()
        # save the output
        if test_save_path is not None:
            if not os.path.exists(test_save_path+"/images/"):
                os.makedirs(test_save_path+"/images/")
            for i in range(image_batch.shape[0]):
                image = image_batch[i].cpu().detach().numpy()
                image = np.squeeze(image)
                image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image.transpose(1, 2, 0))
                image.save(os.path.join(test_save_path+"/images/", case_name + '_img.png'))
                label = label_batch[i]
                label_image = np.stack([np.zeros_like(label)]*3).astype(np.uint8)
                for j in range(1, args.num_classes + 1):
                    label_image[0][label == j] = color_pallete[j][0]
                    label_image[1][label == j] = color_pallete[j][1]
                    label_image[2][label == j] = color_pallete[j][2]
                label_image = Image.fromarray(label_image.transpose(1, 2, 0))
                label_image.save(os.path.join(test_save_path+"/images/", case_name + '_gt.png'))
                # breakpoint()
                pred = prediction[i]
                pred_image = np.stack([np.zeros_like(pred)]*3).astype(np.uint8)
                for j in range(1, args.num_classes + 1):
                    pred_image[0][pred == j] = color_pallete[j][0]
                    pred_image[1][pred == j] = color_pallete[j][1]
                    pred_image[2][pred == j] = color_pallete[j][2]
                pred_image = Image.fromarray(pred_image.transpose(1, 2, 0))
                pred_image.save(os.path.join(test_save_path+"/images/", case_name + '_pred.png'))
        sample_metric_list = []
        for i in range(1, args.num_classes + 1):
            sample_metric_list.append(calculate_metric_percase(prediction == i, label_batch == i))
        
        metric_list.append(np.array(sample_metric_list))
        logger.info('idx %d case %s mean_dice %f' % (
            1, case_name, np.nanmean(sample_metric_list, axis=0)))
    
    metric_list = np.nanmean(metric_list, axis=0)
    for i in range(1, args.num_classes + 1):
        logger.info('Mean class %d name %s mean_dice %f' % (i, class_to_name[i], metric_list[i - 1]))
    
    performance = np.nanmean(metric_list, axis=0)

    logger.info('Testing performance in best val model: mean_dice : %f ' % (performance))
    logger.info("Testing Finished!")


def inference(args, multimask_output, model, test_save_path=None):
    data_fd_list = pd.read_csv(args.data_path+'/test.csv')
    data_fd_list = data_fd_list["image_pth"]
    data_fd_list = [data_fd.split("/")[-3] for data_fd in data_fd_list]
    data_fd_list = list(set(data_fd_list))
    data_fd_list.sort()
    model.cuda()
    model.eval()
    metric_list = []
    # for data_fd in tqdm(data_fd_list):
    for data_fd in data_fd_list:
        image_file_list = os.listdir(args.data_path+'/'+data_fd + '/images')
        image_file_list.sort()
        image_arr_list = []
        mask_arr_list = []
        for image_file in image_file_list:
            with open(args.data_path+'/'+data_fd + '/images/'+image_file, 'rb') as file:
                image_arr = pickle.load(file)
            with open(args.data_path+'/'+data_fd + '/masks/'+image_file.replace("2Dimage", "2Dmask"), 'rb') as file:
                mask_arr = pickle.load(file)

            # image_arr = np.clip(image_arr, HU_min, HU_max)
            # image_arr = (image_arr-HU_min)/(HU_max-HU_min)*255.0
            # image_arr = np.float32(image_arr)
            # image_arr = (image_arr - data_mean) / data_std
            image_arr = (image_arr-image_arr.min())/(image_arr.max()-image_arr.min()+0.00000001)

            mask_arr = np.float32(mask_arr)
            if 'synapseCT' in args.data_path:
                mask_arr[mask_arr==13] = 12
                class_to_name = {1: 'spleen', 2: 'right kidney', 3: 'left kidney', 4: 'gallbladder', 5:'esophagus', 6: 'liver', 7: 'stomach', 8: 'aorta', 9:'vena', 10:'vein', 11: 'pancreas', 12:'adrenal gland'}
            elif 'kits' in args.data_path:
                class_to_name = {1: 'kidney', 2: 'tumor'}
            elif 'prostate' in args.data_path:
                class_to_name = {1: 'prostate'}
            elif 'ixi' in args.data_path:
                class_to_name = {1: 'brain'}
            image_arr_list.append(image_arr)
            mask_arr_list.append(mask_arr)

        image = np.expand_dims(np.stack(image_arr_list), axis=0) 
        label = np.expand_dims(np.stack(mask_arr_list), axis=0)
        case_name = os.path.join(os.path.basename(args.data_path), data_fd)

        h, w = image.shape[2], image.shape[3]

        metric_i = test_single_volume(image, label, model, classes=args.num_classes, 
                                      multimask_output=multimask_output,
                                      patch_size=[args.img_size, args.img_size],
                                      test_save_path=test_save_path, case=case_name)
        
        metric_list.append(np.array(metric_i))
        logger.info('idx %d case %s mean_dice %f' % (
            1, case_name, np.nanmean(metric_i, axis=0)))
    
    metric_list = np.nanmean(metric_list, axis=0)
    for i in range(1, args.num_classes + 1):
        logger.info('Mean class %d name %s mean_dice %f' % (i, class_to_name[i], metric_list[i - 1]))

    performance = np.nanmean(metric_list, axis=0)
    
    logger.info('Testing performance in best val model: mean_dice : %f ' % (performance))
    logger.info("Testing Finished!")
    return 1

def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(': ')
        items_dict[key] = value
    return items_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapt_ckpt', type=str, default='results/Adapter_backbone_frozen/epoch_159.pth', help='The checkpoint after adaptation')
    parser.add_argument('--data_path', type=str, default='data/synapseCT/Training/2D_all_5slice')
    
    parser.add_argument('--num_classes', type=int, default=12)
    parser.add_argument('--img_size', type=int, default=512, help='Input image size of the network')
    
    parser.add_argument('--seed', type=int, default=1234, help='random seed')
    parser.add_argument('--is_savenii', action='store_true', help='Whether to save results during inference')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='pretrained/sam_vit_h_4b8939.pth', help='Pretrained checkpoint')
    parser.add_argument('--vit_name', type=str, default='vit_h', help='Select one vit model')
    parser.add_argument('--rank', type=int, default=32, help='Rank for FacT adaptation')
    parser.add_argument('--scale', type=float, default=1.0)
    parser.add_argument('--num_lora', type=int, default=0, help='Number of LoRA layers')
    parser.add_argument('--federate', action='store_true', help='If activated, use federated learning')
    parser.add_argument('--enable_dora', action='store_true', help='If activated, use DoRA')
    parser.add_argument('--module', type=str, default='sam_fact_tt_image_encoder')

    args = parser.parse_args()

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    args.output_dir = args.adapt_ckpt[:-4]
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                    num_classes=args.num_classes,
                                                                    r=args.rank,
                                                                    lora_scale=args.scale,
                                                                    num_lora=args.num_lora,
                                                                    enable_dora=args.enable_dora,
                                                                    checkpoint=args.ckpt, pixel_mean=[0., 0., 0.],
                                                                pixel_std=[1., 1., 1.])
    
    pkg = import_module(args.module)
    net = pkg.Fact_tt_Sam(sam, args.rank, s=args.scale).cuda()
    # net = sam

    assert args.adapt_ckpt is not None
    _ = net.load_parameters(args.adapt_ckpt)
    if ((args.num_lora > 1) and (args.federate)):
        model_path = args.adapt_ckpt.replace('global_ckpt',f'client_{args.data_path[-1]}_ckpt')
        model_ckpt = torch.load(model_path, map_location='cpu')
        sam_dict = net.state_dict()
        lora_scale_dict = {k: v for k, v in model_ckpt['state_dict'].items() if 'lora_scale' in k}
        sam_dict.update(lora_scale_dict)
        net.load_state_dict(sam_dict)
        print(f'Loaded LoRA scale from {model_path}')  

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False
    
    if not os.path.exists('./results/training_log'):
        os.mkdir('./results/training_log')
    logging.basicConfig(filename= './results/training_log/' + args.adapt_ckpt.split('/')[-2] + '_log.txt',
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logger = logging.getLogger(__name__)#.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)
    logger.info(str(args))
    # model_info = summary(net, input_data=[torch.randn(1, 5, 3, args.img_size, args.img_size), multimask_output, args.img_size],
    #                     col_names=["input_size", "output_size", "num_params", "trainable", "mult_adds"],
    #                     row_settings=["var_names"], depth=15 ,verbose=0)
    if args.is_savenii:
        test_save_path = args.output_dir
    else:
        test_save_path = None
    if 'isic' in args.data_path:
        inference_2D(args, multimask_output, net, test_save_path)
    else:
        inference(args, multimask_output, net, test_save_path)
