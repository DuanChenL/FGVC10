import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import time
import sys

sys.path.append('/data1/PycharmProjects/FGVC10/pytorch-image-models')
import timm

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import config
import json
import csv

from torch.utils.data import Dataset, DataLoader
import cv2
from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout,
    ShiftScaleRotate, CenterCrop, Resize, SmallestMaxSize
)

from albumentations.pytorch import ToTensorV2

from tqdm import tqdm

CFG = {
    'seed': 42,
    'resize_size': 512,
    'img_size': 512,

    'num_classes': 12512,

    'model_arch': ['convnextv2_large.fcmae_ft_in22k_in1k_384',
                   'convnextv2_large.fcmae_ft_in22k_in1k_384'],
                   # 'convnextv2_large.fcmae_ft_in22k_in1k_384'],  # 'swin_base_patch4_window12_384_in22k',
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    # 'mean'=[0.485, 0.456, 0.406],
    # 'std'=[0.229, 0.224, 0.225],
    # mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    # OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    # OPENAI_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    'valid_bs': 32,

    'checkpoints': [
        # 'checkpoints/new_convnextv2_base.fcmae_ft_in22k_in1k_384_mixtype_mixup_mixprob_0.8_seed_42_ls_0.1_epochs_24_diffLR_False.pth'
        # 'checkpoints/new_convnextv2_base.fcmae_ft_in22k_in1k_384_mixtype_cutmix_mixprob_0.8_seed_42_ls_0.1_epochs_24_diffLR_False.pth',
        # 'checkpoints/convnextv2_base.fcmae_ft_in22k_in1k_384_epoch_19_tokenmix_False_p_32_seed_42_ls_0.1_total_20_lr_3.75e-05.pth'
        # 'checkpoints/imgsize_512_convnextv2_base.fcmae_ft_in22k_in1k_384_mixtype_tokenmix_mixprob_0.8_seed_42_ls_0.1_epochs_24_diffLR_False_meta.pth',
        # 'checkpoints/imgsize_512_convnextv2_base.fcmae_ft_in22k_in1k_384_mixtype_cutmix_mixprob_0.8_seed_42_ls_0.1_epochs_24_diffLR_False_meta.pth',
        "checkpoints/imgsize_512_convnextv2_large.fcmae_ft_in22k_in1k_384_mixtype_cutmix_mixprob_0.8_seed_42_ls_0.1_epochs_24_diffLR_False_meta__dropoutBeforeConcat_0.67.pth",
        # "checkpoints/imgsize_512_convnextv2_large.fcmae_ft_in22k_in1k_384_mixtype_cutmix_mixprob_0.8_seed_42_ls_0.1_epochs_24_diffLR_False_meta__dropoutBeforeConcat.pth",
        "checkpoints/imgsize_512_convnextv2_large.fcmae_ft_in22k_in1k_384_mixtype_cutmix_mixprob_0.9_seed_42_ls_0.1_epochs_24_diffLR_False_meta__dropoutBeforeConcat_0.5.pth"
    ],

    'num_workers': 4,
    'device': 'cuda',
    'tta': 5
}


def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    return im_rgb


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_inference_transforms_last():
    return Compose([
        RandomResizedCrop(CFG['img_size'], CFG['img_size'], interpolation=cv2.INTER_CUBIC, scale=(0.8, 1)),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        HueSaturationValue(hue_shift_limit=2, sat_shift_limit=2, val_shift_limit=2, p=0.5),
        RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
        Normalize(mean=CFG['mean'], std=CFG['std'], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)


def get_inference_transforms():
    return Compose([
        SmallestMaxSize(CFG['resize_size'], interpolation=cv2.INTER_CUBIC),
        CenterCrop(CFG['img_size'], CFG['img_size']),
        Transpose(p=0.5),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        HueSaturationValue(hue_shift_limit=1, sat_shift_limit=1, val_shift_limit=1, p=0.5),
        RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
        Normalize(mean=CFG['mean'], std=CFG['std'], max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.)


class InferenceDataset(Dataset):
    def __init__(self, transforms=get_inference_transforms_last()):
        self.transforms = transforms
        self.meta_features_test = np.load(config.PATH['test_meta'])
        test_file = open(config.PATH['test_file'], 'r')
        self.test_file = json.load(test_file)
        test_file.close()

    def __len__(self):
        return len(self.test_file)
        # return 1024

    def label_process(self, label):
        # return torch.tensor([float(x) for x in label])
        return int(label)

    def __getitem__(self, index):
        observation_id = self.label_process(self.test_file[index]['id'])
        img_path = config.PATH['test_img'] + self.test_file[index]['file_name']
        img = get_img(img_path)
        meta_feature = self.meta_features_test[index]
        # print(">>>>>>>>>>>>>>debug")
        # print(img_path)
        # print(meta_feature)

        if self.transforms is not None:
            img = self.transforms(image=img)['image']
        return meta_feature, img, observation_id


def inference(model, dataloader, device=CFG['device']):
    observation_id_lst = []
    probs_lst = []
    model.eval()
    with torch.no_grad():
        for metas, imgs, observation_ids in tqdm(dataloader):
            metas = metas.to(device).float()
            imgs = imgs.to(device).float()
            outs = model(imgs, metas)
            probs = F.softmax(outs)
            for observation_id, prob in zip(observation_ids, probs):
                observation_id_lst.append(observation_id.item())
                probs_lst.append(prob.cpu().numpy())
    probs_lst = np.array(probs_lst)
    return probs_lst, observation_id_lst

class MetaModel(nn.Module):
    def __init__(self, model_arch, feature_dim, meta_feature_dim, num_classes, Flag=True) -> None:
        super().__init__()
        self.backbone = timm.create_model(model_arch, num_classes = 0, pretrained=False)
        self.midd = nn.Linear(feature_dim+meta_feature_dim, 4096)
        self.head = nn.Linear(4096, num_classes)
        self.metaBN = nn.BatchNorm1d(meta_feature_dim)
        if Flag:
            self.dropout = nn.Dropout(p=0.5)
        else:
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

def unique_observation_id(observation_id_lst, array):
    id2count = {}
    id2probs = {}
    for i, obs_id in enumerate(observation_id_lst):
        sum_probs = sum([lst[i] for lst in array])
        avg_obs_id_probs = sum_probs / len(array)
        if obs_id in id2count.keys():
            # print(obs_id)
            id2count[obs_id] += 1
            id2probs[obs_id] += avg_obs_id_probs
        else:
            id2count[obs_id] = 1
            id2probs[obs_id] = avg_obs_id_probs
    for obs_id, count in id2count.items():
        id2probs[obs_id] = id2probs[obs_id] / count
    unique_observation_id_lst, class_id_lst = [], []
    for obs_id, probs in id2probs.items():
        unique_observation_id_lst.append(obs_id)
        label = np.argmax(probs)
        class_id_lst.append(label)
    return unique_observation_id_lst, class_id_lst


def write_csv(observation_id_lst, array, file_name):
    unique_observation_id_lst, class_id_lst = unique_observation_id(observation_id_lst, array)
    csvFile = open(file_name, "w", newline='')  # 创建csv文件
    writer = csv.writer(csvFile)
    writer.writerow(
        ["uniqID", "trait_4", "trait_6", "trait_11", "trait_13", "trait_14", "trait_15", "trait_18", "trait_21",
         "trait_26", "trait_27", "trait_46", "trait_47", "trait_50", "trait_55", "trait_78", "trait_95", "trait_138",
         "trait_144", "trait_145", "trait_146", "trait_163", "trait_169", "trait_223", "trait_224", "trait_237",
         "trait_281", "trait_282", "trait_289", "trait_1080", "trait_3112", "trait_3113", "trait_3114", "trait_3120"])

    with open('data/label2num.json', 'r') as file:
        label2name = json.load(file)
    for i in range(len(unique_observation_id_lst)):
        pred = label2name[str(class_id_lst[i])]
        res = list()
        res.append(str(unique_observation_id_lst[i]))
        for p in pred:
            res.append(p)
        writer.writerow(res)
    csvFile.close()



if __name__ == '__main__':
    array = []

    probs_lst_ensemble = []
    observation_id_lst = []
    num_checkpoint = 0
    for checkpoint in CFG['checkpoints']:
        seed_everything(CFG['seed'])

        print('checkpoint : ', checkpoint)

        dataset = InferenceDataset()
        dataloader = DataLoader(
            dataset,
            batch_size=CFG['valid_bs'],
            num_workers=CFG['num_workers'],
            shuffle=False,
            pin_memory=False,
        )

        device = torch.device(CFG['device'])

        # model = timm.create_model(CFG['model_arch'], num_classes=CFG['num_classes'], pretrained=False)
        # state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))
        # from collections import OrderedDict

        # new_state_dict = OrderedDict()

        # for k, v in state_dict.items():
        #     new_state_dict[k[7:]] = v

        # model.load_state_dict(new_state_dict)
        # model = nn.DataParallel(model)
        # model.to(device)
        
        temp_model = timm.create_model(CFG['model_arch'][num_checkpoint], num_classes = 0, pretrained=False)
        feature_dim = temp_model(torch.rand((1, 3, CFG['img_size'], CFG['img_size']))).shape[1]
        print(feature_dim)
        del temp_model
        meta_feature_dim = np.load(config.PATH['test_meta']).shape[1]
        print(meta_feature_dim)
        flag = True
        if num_checkpoint == 0:
            flag = False
        model = MetaModel(CFG['model_arch'][num_checkpoint], feature_dim, meta_feature_dim, CFG['num_classes'], Flag=flag)
        # print(model)
        
        model_state_dict = model.state_dict()
        
        # for i,(j,k) in enumerate (model_state_dict.items()):
        #     print(j)
        
        state_dict = torch.load(checkpoint, map_location=torch.device('cpu'))
        from collections import OrderedDict
        state_dict = {k[7:]:v for k,v in state_dict.items()}
        # print("debug>>>>>>>>>>>>>>>>>")
        # for i,(j,k) in enumerate (state_dict.items()):
        #     # if i<5:
        #         print(j)
                
        # time.sleep(60)
        
        model.load_state_dict(state_dict)
    
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     new_state_dict[k] = v
        # for k, v in model_state_dict.items():
        #     if k == 'head.fc.weight' or k == 'head.fc.bias':
        #         new_state_dict[k] = v
    
        # model.load_state_dict(new_state_dict)
            
        model = nn.DataParallel(model)
        model.to(device)

        probs_lst = []
        # observation_id_lst = []

        for i in range(CFG['tta']):
            if i == 0:
                probs_lst, observation_id_lst = inference(model, dataloader)
            else:
                t_probs_lst, t_observation_id_lst = inference(model, dataloader)
                for j in range(len(t_probs_lst)):
                    probs_lst[j] += t_probs_lst[j]
        probs_lst = probs_lst / CFG['tta']
        if num_checkpoint == 0:
            probs_lst_ensemble = probs_lst
        else:
            for j in range(len(probs_lst)):
                probs_lst_ensemble[j] += probs_lst[j]
        num_checkpoint += 1
    probs_lst_ensemble = probs_lst_ensemble / num_checkpoint
    array.append(probs_lst_ensemble)

    # write_file_name = './inference/meta_' + os.path.basename(checkpoint[:-4]) + '_cls.csv'
    # write_file_name = './inference/' + os.path.basename('conv_50_imgnet') + '.csv'
    write_file_name = './inference/meta_' + 'ensemble' + '_cls.csv'

    write_csv(observation_id_lst, array, write_file_name)


