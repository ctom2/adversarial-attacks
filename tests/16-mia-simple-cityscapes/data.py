import torch
import os
import numpy as np
from torch.utils import data
import cv2


IMG_DIR='cityscapes/leftImg8bit/train/'
LBL_DIR='cityscapes/gtFine/train/'

def recursive_glob(rootdir=".", suffix=""):
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


def get_cityscapes_paths():
    img_paths = np.sort(os.listdir(IMG_DIR))
    img_paths = recursive_glob(rootdir=IMG_DIR, suffix=".png")

    lbl_paths = []

    for path in img_paths:
        lbl_path = os.path.join(
            LBL_DIR,
            path.split(os.sep)[-2],
            os.path.basename(path)[:-15] + "gtFine_labelIds.png",
        )

        lbl_paths.append(lbl_path)

    lbl_paths = np.array(lbl_paths)

    print('Images not being permuted')
    p = list(np.arange(0, len(img_paths), 1, dtype=int))

    img_paths = np.array(img_paths)
    img_paths = img_paths[p]
    lbl_paths = lbl_paths[p]

    return {'imgs': img_paths, 'lbls': lbl_paths}


def victim_shadow_split(data):
    split = len(data['imgs'])//2

    victim_data = {'imgs': data['imgs'][:split], 'lbls': data['lbls'][:split]}
    shadow_data = {'imgs': data['imgs'][split:], 'lbls': data['lbls'][split:]}

    return victim_data, shadow_data


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


class CityscapesDataset:
    def __init__(self):
        # get the paths to images and labels (masks)
        cs_paths = get_cityscapes_paths()

        # split the paths for the victim and the shadow models
        victim_paths, shadow_paths = victim_shadow_split(cs_paths)

        # split data for training and validation; {'imgs', 'lbls'}
        self.victim_train_paths, self.victim_val_paths = data_split(victim_paths)
        self.shadow_train_paths, self.shadow_val_paths = data_split(shadow_paths)

        # make data for the attack model; {'imgs', 'lbls', 'member'}
        self.victim_attack_paths = attack_data_merge(self.victim_train_paths, self.victim_val_paths)
        self.shadow_attack_paths = attack_data_merge(self.shadow_train_paths, self.shadow_val_paths)


class CityscapesLoader(data.Dataset):
    def __init__(self, data, attack=False):
        self.img_dir = IMG_DIR
        self.lbl_dir = LBL_DIR

        self.img_size = (512,256)
        self.attack = attack

        self.img_paths = data['imgs']
        self.lbl_paths = data['lbls']

        self.n_classes = 19

        if self.attack:
            self.tg = data['member']

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        
        # these are 19
        self.valid_classes = [
            7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33,
        ]
        
        # these are 19 + 1; "unlabelled" is extra
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]
        
        # for void_classes; useful for loss function
        # self.ignore_index = 250
        self.ignore_index = 19
        
        # dictionary of valid classes 7:0, 8:1, 11:2
        self.class_map = dict(zip(self.valid_classes, range(19)))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        # path of image
        # img_path = self.img_dir + self.img_paths[index]
        img_path = self.img_paths[index]

        # path of label
        # lbl_path = self.lbl_dir + self.lbl_paths[index]
        lbl_path = self.lbl_paths[index]

        # read image
        img = cv2.imread(img_path)
        # convert to numpy array
        img = np.array(img, dtype=np.uint8)

        # read label: READ AS GRAYSCALE
        lbl = cv2.imread(lbl_path, 0)
        # encode using encode_segmap function: 0...18 and 250
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        img, lbl = self.transform(img, lbl)

        if self.attack:
            return img, lbl, self.tg[index]

        return img, lbl

    # there are different class 0...33
    # we are converting that info to 0....18; and 250 for void classes
    # final mask has values 0...18 and 250
    def encode_segmap(self, mask):
        # !! Comment in code had wrong informtion
        # Put all void classes to ignore_index
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def transform(self, img, lbl):
        # img = m.imresize(img, (self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
        img = cv2.resize(img, (self.img_size[0], self.img_size[1]))

        # change to BGR
        img = img[:, :, ::-1]  # RGB -> BGR
        # change data type to float64
        img = img.astype(np.float64)
        # subtract mean
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        lbl = cv2.resize(lbl, (self.img_size[0], self.img_size[1]), interpolation = cv2.INTER_NEAREST)
        lbl = lbl.astype(int)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl
