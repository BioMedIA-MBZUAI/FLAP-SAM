import argparse
import logging
import os
import random
import sys
import time
import math
import numpy as np
import torch
import copy
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms
from icecream import ic
from datetime import datetime
from torchinfo import summary
from datasets.dataset import dataset_reader, RandomGenerator
from sklearn.utils.extmath import randomized_svd
# from memory_profiler import profile

debug_lal = False

def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    low_res_logits = outputs['low_res_logits']
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice

class LocalTrainer(object):
    def __init__(self, args, local_model, snapshot_path, multimask_output, low_res, site):
        self.args = args
        self.local_model = local_model
        self.global_model = copy.deepcopy(local_model)
        self.snapshot_path = snapshot_path
        self.multimask_output = multimask_output
        self.low_res = low_res
        self.site = site
        self.base_lr = args.base_lr
        self.num_classes = args.num_classes
        self.batch_size = args.batch_size * args.n_gpu
        base_dir = args.root_path + f'/2D_all_5slice_site{site}'
        self.db_train = dataset_reader(base_dir=base_dir, split="train", num_classes=args.num_classes, 
                                transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size], low_res=[low_res, low_res])]))
        def worker_init_fn(worker_id):
            random.seed(args.seed + worker_id)
        self.trainloader = DataLoader(self.db_train, batch_size=self.batch_size, shuffle=True, 
                             num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
        if args.n_gpu > 1:
            self.local_model = nn.DataParallel(local_model)
        self.local_model.train()
        self.ce_loss = CrossEntropyLoss(ignore_index=-100)
        self.dice_loss = DiceLoss(self.num_classes + 1)
        if args.warmup:
            self.b_lr = self.base_lr / args.warmup_period
        else:
            self.b_lr = self.base_lr
        if args.AdamW:
            self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.local_model.parameters()), lr=self.b_lr, betas=(0.9, 0.999), weight_decay=0.1)
        else:
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.local_model.parameters()), lr=self.b_lr, momentum=0.9, weight_decay=0.0001)
        if args.use_amp:
            self.scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
        else:
            self.scaler = None
        self.writer = SummaryWriter(snapshot_path + '/log')
        self.iter_num = 0
        self.max_iterations = args.max_epochs * len(self.trainloader)
        logging.basicConfig(filename= './results/fed_training_log/' + args.output.split('/')[-1] +f'_{site}_log.txt',
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S', force=True)
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        self.logger = logger
        self.logger.info("{} iterations per epoch. {} max iterations ".format(len(self.trainloader), self.max_iterations))
        self.local_save_mode_path = os.path.join(self.snapshot_path, f'client_{self.site}_ckpt_0.pth')

    def train_one_epoch(self):
        for i_batch, sampled_batch in enumerate(self.trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label'] 
            image_batch = image_batch.unsqueeze(2)
            image_batch = torch.cat((image_batch, image_batch, image_batch), dim=2)
            hw_size = image_batch.shape[-1]
            label_batch = label_batch.contiguous().view(-1, hw_size, hw_size)

            low_res_label_batch = sampled_batch['low_res_label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            low_res_label_batch = low_res_label_batch.cuda()
            proximal_loss = 0
            
            if self.args.use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.args.use_amp):
                    outputs = self.local_model(image_batch, self.multimask_output, self.args.img_size)
                    loss, loss_ce, loss_dice = calc_loss(outputs, label_batch, self.ce_loss, self.dice_loss, self.args.dice_param)

                # check if loss is nan
                if math.isnan(loss.item()):
                    save_path = os.path.join(self.snapshot_path, f'loss_nan_epoch.pth')
                    save_model(self.logger, self.local_model, save_path, 0, self.optimizer, self.scaler)
                    self.logger.info('loss is nan while training...... exiting.....')
                    breakpoint()
                    exit(1)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            else:
                outputs = self.local_model(image_batch, self.multimask_output, self.args.img_size)
                loss, loss_ce, loss_dice = calc_loss(outputs, label_batch, self.ce_loss, self.dice_loss, self.args.dice_param)
                # check if loss is nan
                if math.isnan(loss.item()):
                    save_path = os.path.join(self.snapshot_path, f'loss_nan_epoch_.pth')
                    save_model(self.logger, self.local_model, save_path, 0, self.optimizer)
                    self.logger.info('loss is nan while training...... exiting.....')
                    exit(1)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            if self.args.warmup and self.iter_num < self.args.warmup_period:
                lr_ = self.base_lr * ((self.iter_num + 1) / self.args.warmup_period)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if self.args.warmup:
                    shift_iter = self.iter_num - self.args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = self.iter_num
                lr_ = self.base_lr * (1.0 - shift_iter / self.max_iterations) ** self.args.lr_exp
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr_

            self.iter_num = self.iter_num + 1
            self.writer.add_scalar('info/lr', lr_, self.iter_num)
            self.writer.add_scalar('info/total_loss', loss, self.iter_num)
            self.writer.add_scalar('info/loss_ce', loss_ce, self.iter_num)
            self.writer.add_scalar('info/loss_dice', loss_dice, self.iter_num)

            self.logger.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (self.iter_num, loss.item(), loss_ce.item(), loss_dice.item()))
        
    def save_model(self, epoch_num):
        try:
            state_dict = self.local_model.save_parameters()
        except:
            state_dict = self.local_model.module.save_parameters()
        state = {
            'client': self.site,
            'epoch': epoch_num,
            'iter_num': self.iter_num,
            'state_dict': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict() if self.scaler is not None else None,
        }
        torch.save(state, self.local_save_mode_path)
        self.logger.info("save model of client {} to {}".format(self.site, self.local_save_mode_path))

    def load_model(self, model_pth):
        _ = self.local_model.load_parameters(model_pth) # redundant
        checkpoint = torch.load(self.local_save_mode_path, map_location='cpu')
        self.iter_num = checkpoint['iter_num']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.args.use_amp:
            self.scaler.load_state_dict(checkpoint['scaler'])
        else:
            self.scaler = None
        # move optimizer state to GPU
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        sam_dict = self.local_model.state_dict()
        lora_scale_dict = {k: v for k, v in checkpoint['state_dict'].items() if 'lora_scale' in k}
        sam_dict.update(lora_scale_dict)
        self.local_model.load_state_dict(sam_dict)
        self.logger.info(f"dumping lora_scale_dict: {lora_scale_dict}")
    
    def clear_loggers(self):
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
    
def save_model(logger, model, path, epoch_num, optimizer, scaler=None):
    try:
        state_dict = model.save_parameters()
    except:
        state_dict = model.module.save_parameters()
    state = {
        'epoch': epoch_num,
        'state_dict': state_dict,
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
    }
    torch.save(state, path)
    logger.info("save model to {}".format(path))

def check_loss(loss):
    if math.isnan(loss.item()):
        print('loss is nan while training...... exiting.....')
        return True
    else:
        return False

def trainer_run(args, model, snapshot_path, multimask_output, low_res):
    from datasets.dataset import dataset_reader, RandomGenerator
    
    output_filename = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    if not os.path.exists('./results/training_log'):
        os.mkdir('./results/training_log')
    logging.basicConfig(filename= './results/training_log/' + args.output.split('/')[-1] + '_log.txt',
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logger = logging.getLogger(__name__)#.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)
    logger.info(str(args))
    if args.resume_pth is None:
        # model_info = summary(model, input_data=[torch.randn(args.batch_size, 3, args.img_size, args.img_size).cuda(), multimask_output, args.img_size], # 2D
        model_info = summary(model, input_data=[torch.randn(args.batch_size, 5, 3, args.img_size, args.img_size).cuda(), multimask_output, args.img_size],
                        col_names=["input_size", "output_size", "num_params", "trainable", "mult_adds"],
                        row_settings=["var_names"], depth=15 ,verbose=0)
        logger.info(str(model_info))
        logger.info(str(model))
    # breakpoint()
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    
    db_train = dataset_reader(base_dir=args.root_path, split="train", num_classes=args.num_classes, 
                                transform=transforms.Compose([RandomGenerator(output_size=[args.img_size, args.img_size], low_res=[low_res, low_res])]))
    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, 
                             num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss(ignore_index=-100)
    dice_loss = DiceLoss(num_classes + 1)
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.1)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001) 
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    else:
        scaler = None
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    start_epoch = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)
    logger.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    if args.resume_pth is not None:
        logger.info(f"loading checkpoint from {args.resume_pth}")
        checkpoint = model.load_parameters(args.resume_pth)
        optimizer.load_state_dict(checkpoint['optimizer'])
        if args.use_amp:
            scaler.load_state_dict(checkpoint['scaler'])
        else:
            scaler = None
        start_epoch = checkpoint['epoch'] + 1
        iter_num = start_epoch * len(trainloader)
        logger.info(f"Resuming from epoch {start_epoch}")
    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)
    
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            if 'isic' in args.root_path:
                image_batch = image_batch.unsqueeze(1)
            else:
                image_batch = image_batch.unsqueeze(2)
                image_batch = torch.cat((image_batch, image_batch, image_batch), dim=2)
            hw_size = image_batch.shape[-1]
            label_batch = label_batch.contiguous().view(-1, hw_size, hw_size)

            low_res_label_batch = sampled_batch['low_res_label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            low_res_label_batch = low_res_label_batch.cuda()
            
            if args.use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                    outputs = model(image_batch, multimask_output, args.img_size)
                    loss, loss_ce, loss_dice = calc_loss(outputs, label_batch, ce_loss, dice_loss, args.dice_param)
                # check if loss is nan
                if check_loss(loss):
                    save_mode_path = os.path.join(snapshot_path, f'loss_nan_epoch_{epoch_num}_iter_{iter_num}.pth')
                    save_model(logger, model, save_mode_path, epoch_num, optimizer, scaler)
                    logger.info('loss is nan while training...... exiting.....')
                    exit(1)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                outputs = model(image_batch, multimask_output, args.img_size)
                loss, loss_ce, loss_dice = calc_loss(outputs, label_batch, ce_loss, dice_loss, args.dice_param)
                loss.backward()
                # check if loss is nan
                if check_loss(loss):
                    save_mode_path = os.path.join(snapshot_path, f'loss_nan_epoch_{epoch_num}_iter_{iter_num}.pth')
                    save_model(logger, model, save_mode_path, epoch_num, optimizer)
                    logger.info('loss is nan while training...... exiting.....')
                    exit(1)
                optimizer.step()
                optimizer.zero_grad()
            
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** args.lr_exp
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)

            logger.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

        save_interval = 10 
        if (epoch_num + 1) % save_interval == 0:
            # Saving checkpoint
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            save_model(logger, model, save_mode_path, epoch_num, optimizer, scaler)

        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            save_model(logger, model, save_mode_path, epoch_num, optimizer, scaler)
            iterator.close()
            break

    writer.close()
    return "Training Finished!"

def update_peft_parameters(peft_dict, weights, r=32):
    for key, value in peft_dict.items():
        if ('lora_w_a_' in key):
            A_matrix = value
            B_matrix = peft_dict[key.replace('lora_w_a_', 'lora_w_b_')]

            # AB_matrix = torch.stack([torch.mm(A.T, B.T) for A, B in zip(A_matrix, B_matrix)])
            BA_matrix = torch.stack([w * torch.mm(B, A) for A, B, w in zip(A_matrix, B_matrix, weights)])
            # avg_AB_matrix = torch.mean(AB_matrix, dim=0)
            avg_matrix = torch.sum(BA_matrix, dim=0)

            # U, S, V = torch.linalg.svd(avg_matrix)
            U, S, V = randomized_svd(avg_matrix.cpu().numpy(), n_components=r)
            # peft_dict.update({key.replace('lora_w_a_', 'lora_w_b_'): (U[:,:r]@torch.diag(S[:r]).sqrt())})
            # peft_dict.update({key: (torch.diag(S[:r]).sqrt()@V.t()[:r,:])})
            peft_dict.update({key.replace('lora_w_a_', 'lora_w_b_'): torch.from_numpy(U),
                              key: torch.from_numpy(np.diag(S) @ V)})
        elif ('_FacT' in key):
            raise NotImplementedError
    return peft_dict

def fed_trainer_run(args, global_model, snapshot_path, multimask_output, low_res):
    if not os.path.exists('./results/fed_training_log'):
        os.mkdir('./results/fed_training_log')

    toggle = True
    n_clients = args.num_clients
    client_ids = [chr(65 + i) for i in range(n_clients)]
    start_epoch = 0
    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    if args.resume_pth is not None:
        print(f"loading checkpoint from {args.resume_pth}")
        checkpoint = global_model.load_parameters(args.resume_pth)
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
        global_save_mode_path = args.resume_pth
    
    global_model.cpu()
    # breakpoint()
    for epoch_num in range(start_epoch, stop_epoch):
        for i, client_id in tqdm(enumerate(client_ids), desc=f'Epoch {epoch_num}', total=n_clients):
            local_trainer = LocalTrainer(args=args,
                                        local_model=copy.deepcopy(global_model).cpu(),
                                        snapshot_path=snapshot_path,
                                        multimask_output=multimask_output,
                                        low_res=low_res,
                                        site=client_id,
                                        )
            if (epoch_num == 0):
                if (args.num_lora > 1):
                    lora_state_dict = {}
                    for name, param in local_trainer.local_model.named_parameters():
                        if 'lora_scale' in name:
                            lora_state_dict[name] = param
                            for i in range(args.num_lora):
                                param.data[i] = np.random.rand()
                    local_trainer.logger.info(f"dumping lora_state_dict: {lora_state_dict}")
            else:
                if not toggle:
                    local_trainer.local_save_mode_path = os.path.join(snapshot_path, f'client_{client_id}_ckpt_0.pth')
                else:
                    local_trainer.local_save_mode_path = os.path.join(snapshot_path, f'client_{client_id}_ckpt_1.pth')
                local_trainer.load_model(global_save_mode_path)
            local_trainer.local_model.cuda()
            local_trainer.train_one_epoch()
            if toggle:
                local_trainer.local_save_mode_path = os.path.join(snapshot_path, f'client_{client_id}_ckpt_0.pth')
            else:
                local_trainer.local_save_mode_path = os.path.join(snapshot_path, f'client_{client_id}_ckpt_1.pth')
            local_trainer.save_model(epoch_num)
            local_trainer.clear_loggers()
            local_trainer.local_model.cpu()
            local_trainer = None
            del local_trainer
        
        # load parameters from local models
        parameters = []
        for client_id in client_ids:
            if toggle:
                load_mode_path = os.path.join(snapshot_path, f'client_{client_id}_ckpt_0.pth')
            else:
                load_mode_path = os.path.join(snapshot_path, f'client_{client_id}_ckpt_1.pth')
            parameters.append(torch.load(load_mode_path, map_location='cpu')['state_dict'])
        # aggregate parameters
        global_state_dict = {}
        peft_state_dict = {}
        if 'prostate' in args.root_path:
            weights = [0.15, 0.26, 0.31, 0.08, 0.13, 0.07] #[1/n_clients]*n_clients
        elif 'kits' in args.root_path:  
            weights = [0.03, 0.18, 0.15, 0.10, 0.05, 0.49]
        elif 'synapse' in args.root_path:
            weights = [0.39, 0.32, 0.29]
        elif 'ixi' in args.root_path:
            weights = [0.55, 0.32, 0.13]
        else:
            raise NotImplementedError
        for name, _ in parameters[0].items():
            # global_state_dict[name] = torch.sum(torch.stack([weights[client_id]*parameters[client_id][name] for client_id in range(n_clients)]), dim=0)
            if ('_FacT' not in name) and ('lora_w' not in name) and ('lora_scale' not in name):
                global_state_dict[name] = torch.sum(torch.stack([weights[client_id]*parameters[client_id][name] for client_id in range(n_clients)]), dim=0)
                # global_state_dict[name] = torch.mean(torch.stack([parameters[client_id][name] for client_id in range(n_clients)]), dim=0)
            elif ('lora_scale' not in name):
                peft_state_dict[name] = torch.stack([parameters[client_id][name] for client_id in range(n_clients)])
        peft_state_dict = update_peft_parameters(peft_state_dict, r=args.rank, weights=weights)
        for name, value in peft_state_dict.items():
            global_state_dict[name] = value
        # save global model
        global_save_mode_path = os.path.join(snapshot_path, f'global_ckpt_0.pth')
        state = {
            'epoch': epoch_num,
            'state_dict': global_state_dict,
        }
        torch.save(state, global_save_mode_path)
        if (epoch_num+1) % 20 == 0:
            os.popen(f'cp {global_save_mode_path} {global_save_mode_path.replace("ckpt_0", f"ckpt_{epoch_num}")}')
        global_model.cuda()
        _ = global_model.load_parameters(global_save_mode_path)
        global_model.cpu()
        del parameters, state, global_state_dict
        
        toggle = not toggle
        print(f'next toggle is {toggle}...')
