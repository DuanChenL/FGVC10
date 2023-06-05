import sys

package_paths = [
    "/data1/PycharmProjects/FGVC10/pytorch-image-models",
]
for pth in package_paths:
    sys.path.append(pth)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import config
import json
import torch
import cv2
from albumentations.pytorch import ToTensorV2
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, PiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize, SmallestMaxSize
)
import time
import timm
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD, AdamW, RMSprop
from torch import nn
import random
from tqdm import tqdm
from PIL import Image
import numpy as np
import logging
from sklearn.model_selection import GroupKFold, StratifiedKFold
import pandas as pd


CFG = {
    'seed': 42,  # 719,42,68
    
    # base
    # 'model_arch': 'convnextv2_base.fcmae_ft_in22k_in1k_384',
    # 'checkpoints': './convnextv2_base.inat21_384.pth',
    
    # large
    'model_arch': 'beit_large_patch16_512.in22k_ft_in22k_in1k',  #'convnextv2_large.fcmae_ft_in22k_in1k_384',
    'checkpoints': './convnextv2_large.fcmae_ft_in22k_in1k_384_inat21.pth',
    
    # 'checkpoints': '/data1/PycharmProjects/hufr/project/checkpoints/convnextv2_large.fcmae_ft_in22k_in1k_384_inat21.pth',
    'patch': 16,
    
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    
    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    # mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    # OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
    # OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

    'mix_type': 'cutmix', # cutmix, mixup, tokenmix, none
    'mix_prob': 0.8,

    'img_size': 512,

    'class_num': 12512,

    'warmup_epochs': 1,
    'warmup_lr_factor': 0.01,   # warmup_lr = lr * warmup_lr_factor
    'epochs': 30,

    'train_bs': 16,
    'valid_bs': 128,

    'lr': 1e-4 , # large 7.5e-5  base 1e-5 / 2
    'min_lr': 1e-5 / 7 / 2, # large 1e-6 / 7 / 2  base 1e-5 / 7 / 2

    'differLR': False,
    'bacbone_lr_factor': 0.1,    # if 'differLR' is True, the lr of backbone will be lr * bacbone_lr_factor

    'num_workers': 16 *2 *2,
    'device': 'cuda',
    'smoothing': 0.1,  # label smoothing

    'weight_decay': 1e-4,
    'accum_iter': 1,    # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 1,  # the step of printing loss
}

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler(f"logs/meta_{CFG['model_arch']}_train_cls_dropoutBeforeConcat_0.67.log")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    return im_rgb

import pickle as pkl
class FGVCDataset(Dataset):
    def __init__(self, setname='train',
                 transforms=None,
                 output_label=True,
                 one_hot_label=False):

        super().__init__()
        self.setname = setname
        self.meta_features_train = np.load(config.PATH['train_meta'])
        # print(self.meta_features_train)
        self.meta_features_val = np.load(config.PATH['val_meta'])
        self.transforms = transforms

        self.output_label = output_label
        self.one_hot_label = one_hot_label
        train_file = open(config.PATH['train_file'], 'r')
        val_file = open(config.PATH['val_file'], 'r')
        test_file = open(config.PATH['test_file'], 'r')
        self.train_file = json.load(train_file)
        self.val_file = json.load(val_file)
        self.test_file = json.load(test_file)
        train_file.close()
        val_file.close()
        test_file.close()

    def label_process(self, label):
        # return torch.tensor([float(x) for x in label])
        return int(label)

    def __len__(self):
        if self.setname == 'train':
            # return 256
            return len(self.train_file)
        elif self.setname == 'val':
            # return 256
            return len(self.val_file)
        else:
            return len(self.test_file)

    def __getitem__(self, index: int):
        if self.setname == 'train':
            label = self.label_process(self.train_file[index]['label'])
            img_path = config.PATH['train_img'] + self.train_file[index]['file_name']
            img = get_img(img_path)
            meta_feature = self.meta_features_train[index]
        elif self.setname == 'val':
            label = self.label_process(self.val_file[index]['label'])
            img_path = config.PATH['train_img'] + self.val_file[index]['file_name']
            img = get_img(img_path)
            meta_feature = self.meta_features_val[index]
        else:
            label = self.label_process(self.test_file[index]['id'])
            img_path = config.PATH['test_img'] + self.test_file[index]['file_name']
            img = get_img(img_path)
            meta_feature = None

        if self.transforms:
            img = self.transforms(image=img)['image']
            # img = self.transform(image=img)
            # img = img["image"].transpose(2, 0, 1)
            
        # print(type(meta_feature))
        return meta_feature,img, label


def get_train_transforms():
    return Compose([
        RandomResizedCrop(CFG['img_size'], CFG['img_size'],
                          interpolation=cv2.INTER_CUBIC, scale=(0.5, 1)),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        # ShiftScaleRotate(p=0.3),
        PiecewiseAffine(p=0.5),
        HueSaturationValue(hue_shift_limit=4,
                           sat_shift_limit=4, val_shift_limit=4, p=1.0),
        RandomBrightnessContrast(
            brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=1.0),
        OneOf([
            OpticalDistortion(distort_limit=1.0),
            GridDistortion(num_steps=5, distort_limit=1.),
            # ElasticTransform(alpha=3),
        ], p=0.5),

        Normalize(mean=CFG['mean'], std=CFG['std'],
                  max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)


def get_valid_transforms():
    return Compose([
        # SmallestMaxSize(CFG['img_size']),
        Resize(CFG['img_size'], CFG['img_size'],
               interpolation=cv2.INTER_CUBIC),
        # CenterCrop(CFG['img_size'], CFG['img_size']),
        Normalize(mean=CFG['mean'], std=CFG['std'],
                  max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)


def prepare_dataloader():
    # from catalyst.data.sampler import BalanceClassSampler
    
    train_ds = FGVCDataset(transforms=get_train_transforms(), setname='train', output_label=True,
                           one_hot_label=False)
    valid_ds = FGVCDataset(transforms=get_valid_transforms(), setname='val', output_label=True)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=CFG['train_bs'],
        pin_memory=False,
        drop_last=False,
        shuffle=True,
        num_workers=CFG['num_workers'],
        # sampler=BalanceClassSampler(labels=train_['label'].values, mode="downsampling")
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=CFG['valid_bs'],
        num_workers=CFG['num_workers'],
        shuffle=False,
        pin_memory=False,
    )
    return train_loader, val_loader


class MetaModel(nn.Module):
    def __init__(self, model_arch, feature_dim, meta_feature_dim, num_classes) -> None:
        super().__init__()
        self.backbone = timm.create_model(model_arch, num_classes = 0, pretrained=True)
        self.midd = nn.Linear(feature_dim+meta_feature_dim, 4096)
        self.head = nn.Linear(4096, num_classes)
        self.metaBN = nn.BatchNorm1d(meta_feature_dim)
        self.dropout = nn.Dropout(p=0.67)
        
    def forward(self, x, meta_feature):
        res = self.backbone(x)
        res = self.dropout(res)
        meta_feature = self.metaBN(meta_feature)
        # print(res.shape) #[8,1024]
        outs = torch.cat((res, meta_feature), dim=-1)
        outs = self.midd(outs)
        # outs = self.dropout(outs)
        outs = self.head(outs)
        return outs

def train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, device, scheduler=None, schd_batch_update=False):
    model.train()

    running_loss = None
    image_preds_all = []
    image_targets_all = []

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (imgs, image_labels) in pbar:
        metas = metas.to(device).float()
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        with autocast():
            image_preds = model(imgs)

            image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
            image_targets_all += [image_labels.detach().cpu().numpy()]

            loss = loss_fn(image_preds, image_labels)
            loss.duoble()

            scaler.scale(loss).backward()

            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = running_loss * .99 + loss.item() * .01

            if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):
                # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if scheduler is not None and schd_batch_update:
                    scheduler.step()

            if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(train_loader)):
                description = f'epoch {epoch} loss: {running_loss:.4f}'
                pbar.set_description(description)

    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    accuracy = (image_preds_all == image_targets_all).mean()
    print('Train multi-class accuracy = {:.4f}'.format(accuracy))
    logger.info(' Epoch: ' + str(epoch) + ' Train multi-class accuracy = {:.4f}'.format(accuracy))
    logger.info(' Epoch: ' + str(epoch) + ' Train loss = {:.4f}'.format(running_loss))

    if scheduler is not None and not schd_batch_update:
        scheduler.step()
        
    return accuracy


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def generate_mask_random(imgs, patch=CFG['patch'], mask_token_num_start=14, lam=0.5):
    _, _, W, H = imgs.shape
    assert W % patch == 0
    assert H % patch == 0
    p = W // patch
    
    mask_ratio = 1 - lam
    num_masking_patches = min(p**2, int(mask_ratio * (p**2)) + mask_token_num_start)
    mask_idx = np.random.permutation(p**2)[:num_masking_patches] 
    lam = 1 - num_masking_patches / (p**2)
    return mask_idx, lam


def get_mixed_data(imgs, image_labels, mix_type):
    assert mix_type in ['mixup', 'cutmix', 'tokenmix']
    if mix_type == 'mixup':
        alpha = 2.0
        rand_index = torch.randperm(imgs.size()[0]).cuda()
        target_a = image_labels
        target_b = image_labels[rand_index]
        lam = np.random.beta(alpha, alpha)
        imgs = imgs * lam + imgs[rand_index] * (1 - lam)
    elif mix_type == 'cutmix':
        beta = 1.0
        lam = np.random.beta(beta, beta)
        rand_index = torch.randperm(imgs.size()[0]).cuda()
        target_a = image_labels
        target_b = image_labels[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
        imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))
    elif mix_type == 'tokenmix':
        B, C, W, H = imgs.shape
        mask_idx, lam = generate_mask_random(imgs)
        rand_index = torch.randperm(imgs.size()[0]).cuda()
        p = W // CFG['patch']
        patch_w = CFG['patch']
        patch_h = CFG['patch']
        for idx in mask_idx:
            row_s = idx // p
            col_s = idx % p
            x1 = patch_w * row_s
            x2 = x1 + patch_w
            y1 = patch_h * col_s
            y2 = y1 + patch_h
            imgs[:, :, x1:x2, y1:y2] = imgs[rand_index, :, x1:x2, y1:y2]
        
        target_a = image_labels
        target_b = image_labels[rand_index]
        
    return imgs, target_a, target_b, lam


def train_one_epoch_mix(epoch, model, loss_fn, optimizer, train_loader, device, scheduler=None, schd_batch_update=False, mix_type=CFG['mix_type']):
    model.train()

    running_loss = None
    image_preds_all = []
    image_targets_all = []

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (metas, imgs, image_labels) in pbar:
        metas = metas.to(device).float()
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        if np.random.rand(1) < CFG['mix_prob']:
            imgs, target_a, target_b, lam = get_mixed_data(imgs, image_labels, mix_type)
            with autocast():
                image_preds = model(imgs, metas)
                loss = loss_fn(image_preds, target_a) * lam + loss_fn(image_preds, target_b) * (1. - lam)
                loss = loss
                scaler.scale(loss).backward()
        else:
            with autocast():
                image_preds = model(imgs, metas)
                loss = loss_fn(image_preds, image_labels)
                loss = loss
                scaler.scale(loss).backward()
        
        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]

        if running_loss is None:
            running_loss = loss.item()
        else:
            running_loss = running_loss * .99 + loss.item() * .01

        if ((step + 1) % CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):
            # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None and schd_batch_update:
                scheduler.step()

        if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(train_loader)):
            description = f'epoch {epoch} loss: {running_loss:.4f}'
            pbar.set_description(description)

    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    accuracy = (image_preds_all == image_targets_all).mean()

    print('Train multi-class accuracy = {:.4f}'.format(accuracy))
    logger.info(' Epoch: ' + str(epoch) + ' Train multi-class accuracy = {:.4f}'.format(accuracy))
    logger.info(' Epoch: ' + str(epoch) + ' Train loss = {:.4f}'.format(running_loss))

    if scheduler is not None and not schd_batch_update:
        scheduler.step()
        
    return accuracy


def valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False):
    model.eval()

    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []

    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (metas, imgs, image_labels) in pbar:
        metas = metas.to(device).float()
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        image_preds = model(imgs,metas)
        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]
        # if openset, transform labels to calculate loss without errors
        openset_idx = image_labels == -1
        image_labels[openset_idx] = 0   # just assign class_id: 0
        loss = loss_fn(image_preds, image_labels)

        loss_sum += loss.item() * image_labels.shape[0]
        sample_num += image_labels.shape[0]

        if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(val_loader)):
            description = f'epoch {epoch} loss: {loss_sum / sample_num:.4f}'
            pbar.set_description(description)

    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)

    accuracy = (image_preds_all == image_targets_all).mean()
    print('validation multi-class accuracy = {:.4f}'.format(accuracy))
    logger.info(' Epoch: ' + str(epoch) + 'validation multi-class accuracy = {:.4f}'.format(accuracy))

    if scheduler is not None:
        if schd_loss_update:
            scheduler.step(loss_sum / sample_num)
        else:
            scheduler.step()
    return accuracy


if __name__ == '__main__':
    seed_everything(CFG['seed'])
    

    device = torch.device(CFG['device'])
    
    temp_model = timm.create_model(CFG['model_arch'], num_classes=0, pretrained=False)
    feature_dim = temp_model(torch.rand((1,3,CFG['img_size'],CFG['img_size']))).shape[1]
    print(feature_dim) #12512
    del temp_model
    meta_feature_dim = np.load(config.PATH['train_meta']).shape[1]
    print(meta_feature_dim)  #17
    
    model = MetaModel(CFG['model_arch'], feature_dim, meta_feature_dim, CFG['class_num'])

    
    # model = timm.create_model(CFG['model_arch'], num_classes=CFG['class_num'], pretrained=True)
    # model_state_dict = model.state_dict()
    # model = convnextv2_base(num_classes=CFG['num_classes'])
    # state_dict = torch.load(CFG['checkpoints'], map_location=torch.device('cpu'))
    # from collections import OrderedDict
    # print(state_dict)
    
    # base mode
    # state_dict = {'backbone.'+k:v for k,v in state_dict.items()}
    # model.load_state_dict(state_dict,strict=False)
    
    # large mode
    # state_dict = {'backbone.'+k[7:]:v for k,v in state_dict.items()}
    # state_dict = {'lyy.'+k if k.startswith('module.head') else 'backbone.'+k[7:]:v for k,v in state_dict.items()}
    # model.load_state_dict(state_dict,strict=False)
    # print(model_state_dict)
        
    model.to(device)

    model = nn.DataParallel(model)
    
    train_loader, val_loader = prepare_dataloader()

    scaler = GradScaler()

    # set different learning rate for backbone and head
    head_params = list(map(id, model.module.head.parameters()))
    backbone_params = filter(lambda p: id(p) not in head_params, model.parameters())
    lr_cfg = [ {'params': backbone_params, 'lr': CFG['lr'] * CFG['bacbone_lr_factor']},
                {'params': model.module.head.parameters(), 'lr': CFG['lr']}]
    
    if CFG['differLR']:
        optimizer = torch.optim.AdamW(lr_cfg, lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])


    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG['epochs'] - CFG['warmup_epochs'], eta_min=CFG['min_lr']
    )
    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=CFG['warmup_lr_factor'], total_iters=CFG['warmup_epochs']
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[CFG['warmup_epochs']]
    )

    loss_tr = nn.CrossEntropyLoss(label_smoothing=CFG['smoothing']).to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=CFG['smoothing']).to(device)

    best_acc = 0.0
    beat_train_acc = 0.0
    for epoch in range(CFG['epochs']):
        print(optimizer.param_groups[0]['lr'])
        
        temp_train_acc = 0.0

        if CFG['mix_type'] == 'none':
            temp_train_acc = train_one_epoch(epoch, model, loss_tr, optimizer, train_loader, device, scheduler=scheduler)
        else:
            temp_train_acc = train_one_epoch_mix(epoch, model, loss_tr, optimizer, train_loader, device, scheduler=scheduler)
        
        # temp_acc = 0.0
        with torch.no_grad():
            # temp_acc = valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False)
            if (temp_train_acc > beat_train_acc):
                torch.save(model.state_dict(), './checkpoints/imgsize_{}_{}_mixtype_{}_mixprob_{}_seed_{}_ls_{}_epochs_{}_diffLR_{}_meta__dropoutBeforeConcat_0.67.pth'.format(
                                                CFG['img_size'],
                                                CFG['model_arch'],
                                                CFG['mix_type'],
                                                CFG['mix_prob'],
                                                CFG['seed'],
                                                CFG['smoothing'],
                                                CFG['epochs'],
                                                CFG['differLR']))
        if temp_train_acc > beat_train_acc:
            beat_train_acc = temp_train_acc
    
    del model, optimizer, train_loader, val_loader, scaler, scheduler
    print(beat_train_acc)
    logger.info('BEST-train-ACC: ' + str(beat_train_acc))
    torch.cuda.empty_cache()
