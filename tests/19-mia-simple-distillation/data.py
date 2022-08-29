# -----------------------------------
# TOTAL PATHS FOR THE DATASET IS 5237

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


def get_liver_paths():
    img_paths = np.random.permutation(np.sort(os.listdir(IMG_DIR)))
    lbl_paths = img_paths
    return {'imgs': img_paths, 'lbls': lbl_paths}


# -----------------------------------------------------------------------------------------------

class LiverDataset:
    def __init__(self):
        # get the paths to images and labels (masks)
        liver_paths = get_liver_paths()

        # ========================
        # MANUAL DATA SPLIT
        # limiting the size of the whole dataset
        SPLIT_BOUNDARY = 5000

        liver_paths['imgs'] =  liver_paths['imgs'][:SPLIT_BOUNDARY]
        liver_paths['lbls'] =  liver_paths['lbls'][:SPLIT_BOUNDARY]

        # splitting the dataset into victim and shadow data
        VS_SPLIT = SPLIT_BOUNDARY//2 # 2500

        victim_data = {'imgs': liver_paths['imgs'][:VS_SPLIT], 'lbls': liver_paths['lbls'][:VS_SPLIT]}
        shadow_data = {'imgs': liver_paths['imgs'][VS_SPLIT:], 'lbls': liver_paths['lbls'][VS_SPLIT:]}

        # making training and validation sets
        TV_SPLIT = int((VS_SPLIT//5) * 4) # makes 2000/500 split

        self.victim_train_paths = {'imgs': victim_data['imgs'][:TV_SPLIT], 'lbls': victim_data['lbls'][:TV_SPLIT]}
        self.victim_val_paths = {'imgs': victim_data['imgs'][TV_SPLIT:], 'lbls': victim_data['lbls'][TV_SPLIT:]}

        self.shadow_train_paths = {'imgs': shadow_data['imgs'][:TV_SPLIT], 'lbls': shadow_data['lbls'][:TV_SPLIT]}
        self.shadow_val_paths = {'imgs': shadow_data['imgs'][TV_SPLIT:], 'lbls': shadow_data['lbls'][TV_SPLIT:]}

        # ++++++++++++++++++++++++++
        # artifical limiting of the training datatsets, has to be >= 500

        TRAIN_LIM=500

        # making the reference dataset
        self.pre_reference_paths = {
            'imgs': self.victim_train_paths['imgs'][TRAIN_LIM:3*TRAIN_LIM], 
            'lbls': self.victim_train_paths['lbls'][TRAIN_LIM:3*TRAIN_LIM]
        }

        self.victim_train_paths['imgs'] = self.victim_train_paths['imgs'][:TRAIN_LIM]
        self.victim_train_paths['lbls'] = self.victim_train_paths['lbls'][:TRAIN_LIM]

        self.shadow_train_paths['imgs'] = self.shadow_train_paths['imgs'][:TRAIN_LIM]
        self.shadow_train_paths['lbls'] = self.shadow_train_paths['lbls'][:TRAIN_LIM]

        # ++++++++++++++++++++++++++

        # making the attack datasets

        A_SPLIT = VS_SPLIT - TV_SPLIT

        self.victim_attack_paths = {
            'imgs': np.concatenate([self.victim_train_paths['imgs'][:A_SPLIT], self.victim_val_paths['imgs']]),
            'lbls': np.concatenate([self.victim_train_paths['lbls'][:A_SPLIT], self.victim_val_paths['lbls']]),
            'member': np.concatenate([np.ones((A_SPLIT)), np.zeros((A_SPLIT))])
        }

        self.shadow_attack_paths = {
            'imgs': np.concatenate([self.shadow_train_paths['imgs'][:A_SPLIT], self.shadow_val_paths['imgs']]),
            'lbls': np.concatenate([self.shadow_train_paths['lbls'][:A_SPLIT], self.shadow_val_paths['lbls']]),
            'member': np.concatenate([np.ones((A_SPLIT)), np.zeros((A_SPLIT))])
        }

        # ========================

        print('************************')
        print('Victim train paths:', len(self.victim_train_paths['imgs']))
        print('Shadow train paths:', len(self.shadow_train_paths['imgs']))
        print('Attack train paths:', len(self.shadow_attack_paths['imgs']))
        print('Attack val paths:', len(self.victim_attack_paths['imgs']))
        print('************************')
        

# -----------------------------------------------------------------------------------------------

class LiverLoader(data.Dataset):
    def __init__(self, data, attack=False):
        self.img_dir = IMG_DIR
        self.lbl_dir = LBL_DIR

        self.img_size = (256,256)
        self.attack = attack

        self.img_paths = data['imgs']
        self.lbl_paths = data['lbls']

        if self.attack:
            self.tg = data['member']

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
        img = cv2.resize(img, (self.img_size[0], self.img_size[1]))
        img = img.astype(np.float64)
        
        lbl = lbl.astype(float)
        lbl = cv2.resize(lbl, (self.img_size[0], self.img_size[1]), interpolation = cv2.INTER_NEAREST)
        lbl = lbl.astype(int)
        lbl = np.clip(lbl, 0, 1)

        img = torch.from_numpy(np.array([img])).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl