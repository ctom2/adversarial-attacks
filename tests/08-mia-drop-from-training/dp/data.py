import os
import cv2
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader
import SimpleITK as sitk

np.random.seed(2)
IMG_DIR='liver/imgs/'
LBL_DIR='liver/lbls/'


def getNii(path):
    data = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(data)


def get_liver_paths():
    img_paths = np.random.permutation(np.sort(os.listdir(IMG_DIR)))
    lbl_paths = img_paths
    return {'imgs': img_paths, 'lbls': lbl_paths}


def data_split(data):
    imgs = data['imgs']
    lbls = data['lbls']

    split = (len(imgs)//4)*3

    train_data = {'imgs': imgs[:split], 'lbls': lbls[:split]}
    val_data = {'imgs': imgs[split:], 'lbls': lbls[split:]}

    return train_data, val_data

# make balanced in/out attack data
def attack_data_merge(train_data, val_data):
    split = len(val_data['imgs'])
    train_data['imgs'] = train_data['imgs'][:split]
    train_data['lbls'] = train_data['lbls'][:split]

    train_data['member'] = np.ones(len(train_data['imgs']))
    val_data['member'] = np.zeros(len(val_data['imgs']))

    attack_data = {
        'imgs': np.concatenate([train_data['imgs'], val_data['imgs']]),
        'lbls': np.concatenate([train_data['lbls'], val_data['lbls']]),
        'member': np.concatenate([train_data['member'], val_data['member']]),
    }

    return attack_data


def victim_shadow_split(data):
    split = len(data['imgs'])//2

    victim_data = {'imgs': data['imgs'][:split], 'lbls': data['lbls'][:split]}
    shadow_data = {'imgs': data['imgs'][split:], 'lbls': data['lbls'][split:]}

    return victim_data, shadow_data

# removes a portion of the training data (to evalute overfitting)
def remove_training_data(data):
    set = {'imgs': data['imgs'][:50], 'lbls': data['lbls'][:50]}
    train_set = {'imgs': data['imgs'][50:], 'lbls': data['lbls'][50:]}
    return set, train_set

# -----------------------------------------------------------------------------------------------

class LiverDataset:
    def __init__(self):
        # get the paths to images and labels (masks)
        liver_paths = get_liver_paths()

        # split the paths for the victim and the shadow models
        victim_paths, shadow_paths = victim_shadow_split(liver_paths)

        # split data for training and validation; {'imgs', 'lbls'}
        self.victim_train_paths, self.victim_val_paths = data_split(victim_paths)
        self.shadow_train_paths, self.shadow_val_paths = data_split(shadow_paths)

        # make data for the attack model; {'imgs', 'lbls', 'member'}
        self.victim_attack_paths = attack_data_merge(self.victim_train_paths, self.victim_val_paths)
        self.shadow_attack_paths = attack_data_merge(self.shadow_train_paths, self.shadow_val_paths)

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
