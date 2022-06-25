import os
import cv2
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import SimpleITK as sitk

IMG_DIR='liver/imgs/'
LBL_DIR='liver/lbls/'

def getNii(path):
    data = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(data)

class LiverDataset:
    def __init__(self):
        self.img_dir = IMG_DIR
        self.lbl_dir = LBL_DIR

        np.random.permutation(2)
        self.img_paths = np.random.permutation(np.sort(os.listdir(self.img_dir)))
        self.lbl_paths = self.img_paths

        # splitting the paths for training/validating the victim/shadow model
        s = len(self.img_paths)//2
        self.victim_img_paths = self.img_paths[:s]
        self.victim_lbl_paths = self.lbl_paths[:s]
        self.shadow_img_paths = self.img_paths[s:]
        self.shadow_lbl_paths = self.lbl_paths[s:]

        victim_split = (len(self.victim_img_paths)//4)*3
        shadow_split = (len(self.shadow_img_paths)//4)*3

        # for victim/shadow model training/testing

        self.victim_data_train = {
            'images' : self.victim_img_paths[:victim_split],
            'labels' : self.victim_lbl_paths[:victim_split],
        }

        self.victim_data_val = {
            'images' : self.victim_img_paths[victim_split:],
            'labels' : self.victim_lbl_paths[victim_split:],
        }

        self.shadow_data_train = {
            'images' : self.shadow_img_paths[:shadow_split],
            'labels' : self.shadow_lbl_paths[:shadow_split],
        }

        self.shadow_data_val = {
            'images' : self.shadow_img_paths[shadow_split:],
            'labels' : self.shadow_lbl_paths[shadow_split:],
        }

        print('Victim train images:', len(self.victim_data_train['images']))
        print('Victim train labels:', len(self.victim_data_train['labels']))
        print('Victim val images:', len(self.victim_data_val['images']))
        print('Victim val labels:', len(self.victim_data_val['labels']))
        print('Shadow train images:', len(self.shadow_data_train['images']))
        print('Shadow train labels:', len(self.shadow_data_train['labels']))
        print('Shadow val images:', len(self.shadow_data_val['images']))
        print('Shadow val labels:', len(self.shadow_data_val['labels']))

        # for attack model training/testing

        attack_shadow_img = np.concatenate([self.shadow_data_train['images'][:len(self.shadow_data_val['images'])], self.shadow_data_val['images']])
        attack_shadow_lbl = np.concatenate([self.shadow_data_train['labels'][:len(self.shadow_data_val['labels'])], self.shadow_data_val['labels']])
        attack_shadow_tg = np.concatenate([np.ones(len(self.shadow_data_val['images'])), np.zeros(len(self.shadow_data_val['images']))])

        p = np.random.permutation(range(len(attack_shadow_img)))
        attack_shadow_img = attack_shadow_img[p]
        attack_shadow_lbl = attack_shadow_lbl[p]
        attack_shadow_tg = attack_shadow_tg[p]

        attack_shadow_split = (len(attack_shadow_img)//5)*4

        self.attack_data_train = {
            'images' : attack_shadow_img[:attack_shadow_split],
            'labels' : attack_shadow_lbl[:attack_shadow_split],
            'targets' : attack_shadow_tg[:attack_shadow_split],
        }

        self.attack_data_val = {
            'images' : attack_shadow_img[attack_shadow_split:],
            'labels' : attack_shadow_lbl[attack_shadow_split:],
            'targets' : attack_shadow_tg[attack_shadow_split:],
        }

        attack_victim_img = np.concatenate([self.victim_data_train['images'][:len(self.victim_data_val['images'])], self.victim_data_val['images']])
        attack_victim_lbl = np.concatenate([self.victim_data_train['labels'][:len(self.victim_data_val['labels'])], self.victim_data_val['labels']])
        attack_victim_tg = np.concatenate([np.ones(len(self.victim_data_val['images'])), np.zeros(len(self.victim_data_val['images']))])

        self.attack_data_test = {
            'images' : attack_victim_img,
            'labels' : attack_victim_lbl,
            'targets' : attack_victim_tg,
        }

        print('Attack train images:', len(self.attack_data_train['images']))
        print('Attack train labels:', len(self.attack_data_train['labels']))
        print('Attack val images:', len(self.attack_data_val['images']))
        print('Attack val labels:', len(self.attack_data_val['labels']))
        print('Attack test images:', len(self.attack_data_test['images']))
        print('Attack test labels:', len(self.attack_data_test['labels']))

# ************************************************************************************************

class LiverLoader(data.Dataset):
    def __init__(self, data, attack=False):
        self.img_dir = IMG_DIR
        self.lbl_dir = LBL_DIR

        self.img_size = (256,256)
        self.attack = attack

        self.img_paths = data['images']
        self.lbl_paths = data['labels']

        if self.attack:
            self.tg = data['targets']

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        # path of image
        img_path = self.img_dir + self.img_paths[index]

        # path of label
        lbl_path = self.lbl_dir + self.lbl_paths[index]

        # read image
        img = getNii(img_path)
        img = np.array(img, dtype=np.uint8)

        # read label
        lbl = getNii(lbl_path)
        
        img, lbl = self.transform(img, lbl)

        if self.attack:
            return img, lbl, self.tg[index]

        return img, lbl

    def transform(self, img, lbl):
        # img = (img - np.min(img))/(np.max(img) - np.min(img))
        img = cv2.resize(img, (self.img_size[0], self.img_size[1]))
        img = img.astype(np.float64)
        # NHWC -> NCHW
        # img = img.transpose(2, 0, 1)
        
        lbl = lbl.astype(float)
        lbl = cv2.resize(lbl, (self.img_size[0], self.img_size[1]), interpolation = cv2.INTER_NEAREST)
        lbl = lbl.astype(int)
        lbl = np.clip(lbl, 0, 1)

        img = torch.from_numpy(np.array([img])).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

# ************************************************************************************************

class MIAdata:
    def __init__(self, segmentation_batch_size=4, attack_batch_size=4):

        self.segmentation_batch_size = segmentation_batch_size
        self.attack_batch_size = attack_batch_size

        self.kvasir_dataset = LiverDataset()

        # datasets

        self.victim_dataset_train = LiverLoader(self.kvasir_dataset.victim_data_train)
        self.victim_dataset_val = LiverLoader(self.kvasir_dataset.victim_data_val)

        self.shadow_dataset_train = LiverLoader(self.kvasir_dataset.shadow_data_train)
        self.shadow_dataset_val = LiverLoader(self.kvasir_dataset.shadow_data_val)

        self.attack_dataset_train = LiverLoader(self.kvasir_dataset.attack_data_train, attack=True)
        self.attack_dataset_val = LiverLoader(self.kvasir_dataset.attack_data_val, attack=True)
        self.attack_dataset_test = LiverLoader(self.kvasir_dataset.attack_data_test, attack=True)

        # dataloaders

        self.victim_dataloader_train = DataLoader(self.victim_dataset_train, batch_size=self.segmentation_batch_size, shuffle=True)
        self.victim_dataloader_val = DataLoader(self.victim_dataset_val, batch_size=4, shuffle=True)

        self.shadow_dataloader_train = DataLoader(self.shadow_dataset_train, batch_size=self.segmentation_batch_size, shuffle=True)
        self.shadow_dataloader_val = DataLoader(self.shadow_dataset_val, batch_size=4, shuffle=True)

        self.attack_dataloader_train = DataLoader(self.attack_dataset_train, batch_size=self.attack_batch_size, shuffle=True)
        self.attack_dataloader_val = DataLoader(self.attack_dataset_val, batch_size=4, shuffle=True)
        self.attack_dataloader_test = DataLoader(self.attack_dataset_test, batch_size=4, shuffle=True)
        