import torch
from torch.utils import data
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import models

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import opacus

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ************************************************************************************************

class KvasirDataset:
    def __init__(self):
        self.img_dir = 'Kvasir-SEG/images/'
        self.lbl_dir = 'Kvasir-SEG/masks/'

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

class KvasirLoader(data.Dataset):
    def __init__(self, data, attack=False):
        self.img_dir = 'Kvasir-SEG/images/'
        self.lbl_dir = 'Kvasir-SEG/masks/'

        self.img_size = (512,512)
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
        img = cv2.imread(img_path)
        # convert to numpy array
        img = np.array(img, dtype=np.uint8)

        # read label: READ AS GRAYSCALE
        lbl = cv2.imread(lbl_path, 0)
        
        img, lbl = self.transform(img, lbl)

        if self.attack:
            return img, lbl, self.tg[index]

        return img, lbl

    def transform(self, img, lbl):
        img = cv2.resize(img, (self.img_size[0], self.img_size[1]))
        img = img.astype(np.float64)
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)
        
        lbl = lbl.astype(float)
        lbl = cv2.resize(lbl, (self.img_size[0], self.img_size[1]), interpolation = cv2.INTER_NEAREST)
        lbl = lbl.astype(int)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl/255).long()

        return img, lbl

# ************************************************************************************************

class MIAdata:
    def __init__(self, segmentation_batch_size=4, attack_batch_size=4):

        self.segmentation_batch_size = segmentation_batch_size
        self.attack_batch_size = attack_batch_size

        self.kvasir_dataset = KvasirDataset()

        # datasets

        self.victim_dataset_train = KvasirLoader(self.kvasir_dataset.victim_data_train)
        self.victim_dataset_val = KvasirLoader(self.kvasir_dataset.victim_data_val)

        self.shadow_dataset_train = KvasirLoader(self.kvasir_dataset.shadow_data_train)
        self.shadow_dataset_val = KvasirLoader(self.kvasir_dataset.shadow_data_val)

        self.attack_dataset_train = KvasirLoader(self.kvasir_dataset.attack_data_train, attack=True)
        self.attack_dataset_val = KvasirLoader(self.kvasir_dataset.attack_data_val, attack=True)
        self.attack_dataset_test = KvasirLoader(self.kvasir_dataset.attack_data_test, attack=True)

        # dataloaders

        self.victim_dataloader_train = DataLoader(self.victim_dataset_train, batch_size=self.segmentation_batch_size, shuffle=True)
        self.victim_dataloader_val = DataLoader(self.victim_dataset_val, batch_size=self.segmentation_batch_size, shuffle=True)

        self.shadow_dataloader_train = DataLoader(self.shadow_dataset_train, batch_size=self.segmentation_batch_size, shuffle=True)
        self.shadow_dataloader_val = DataLoader(self.shadow_dataset_val, batch_size=self.segmentation_batch_size, shuffle=True)

        self.attack_dataloader_train = DataLoader(self.attack_dataset_train, batch_size=self.attack_batch_size, shuffle=True)
        self.attack_dataloader_val = DataLoader(self.attack_dataset_val, batch_size=self.attack_batch_size, shuffle=True)
        self.attack_dataloader_test = DataLoader(self.attack_dataset_test, batch_size=self.attack_batch_size, shuffle=True)

# ************************************************************************************************

# training of a segmentation model
def segmentation_train(train_dataloader, val_dataloader, epochs=5, lr=2e-4, encoder='resnet18', dp=False):
    
    model = smp.Unet(
        encoder_name=encoder, 
        in_channels=3, 
        classes=1,
    ).to(device)

    criterion = smp.losses.DiceLoss('binary')

    if dp:
        model = opacus.validators.ModuleValidator.fix(model).to(device)
        print('Opacus validation:', opacus.validators.ModuleValidator.validate(model, strict=True))
        
        privacy_engine = opacus.PrivacyEngine()

        opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
        model, opt, train_dataloader = privacy_engine.make_private(
            module=model,
            optimizer=opt,
            data_loader=train_dataloader,
            noise_multiplier=1.1,
            max_grad_norm=0.1,
        )
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))


    for epoch in range(epochs):
        print('EPOCH', epoch, '--------------------')

        train_loss_data = []
        val_loss_data = []

        # Training loop
        model.train()
        for img, lbl in train_dataloader:
            img, lbl = img.to(device), lbl.to(device)
            lbl = lbl.view(img.shape[0],1,img.shape[2],img.shape[3])

            opt.zero_grad()
            pred = model(img)

            loss = criterion(pred.float(), lbl.float())

            loss.backward()
            opt.step()

            train_loss_data.append(loss.item())

            # Training results
            if len(train_loss_data) % 10 == 0:
                print(
                    'training loss:', round(np.sum(np.array(train_loss_data))/len(train_loss_data),4)
                )

        # Validation loop
        model.eval()
        with torch.no_grad():
            for img, lbl in val_dataloader:

                img, lbl = img.to(device), lbl.to(device)
                lbl = lbl.view(img.shape[0],1,img.shape[2],img.shape[3])

                pred = model(img)

                loss = criterion(pred.float(), lbl.float())

                val_loss_data.append(loss.item())

        # Validation results
        print(
            ' > Validation loss:', round(np.sum(np.array(val_loss_data))/len(val_loss_data),4)
        )

        plt.figure(figsize=(10,10))
        plt.subplot(1,3,1)
        plt.imshow(img[0].detach().cpu().numpy().transpose(1,2,0).astype(int))
        plt.subplot(1,3,2)
        plt.imshow(lbl[0,0].detach().cpu().numpy().astype(int))
        plt.subplot(1,3,3)
        plt.imshow(pred[0].view(img.shape[2],img.shape[3]).detach().cpu().numpy(), vmin=0, vmax=1)
        plt.show()

    return model

# ************************************************************************************************

def attacker_train(attack_model, shadow_model, train_dataloader, opt, criterion, attack_2ch=False, argmax=False):
    pred_labels = np.array([])
    true_labels = np.array([])

    # Training loop
    attack_model.train()
    for data, labels, targets in train_dataloader:
        data, labels, targets = data.to(device), labels.to(device), targets.to(device)

        opt.zero_grad()

        s_output = shadow_model(data).view(data.shape[0],1,data.shape[2],data.shape[3])

        if attack_2ch:
            cat = labels.view(data.shape[0],1,data.shape[2],data.shape[3])
            s_output = torch.concat((s_output, cat), dim=1)

        if argmax:
            s_output = torch.round(s_output)

        output = attack_model(s_output)

        loss = criterion(output.float(), targets.float().view(len(targets),1))

        loss.backward()
        opt.step()

        pred_l = output.float().round().view(data.shape[0]).detach().cpu().numpy()
        true_l = targets.float().view(data.shape[0]).detach().cpu().numpy()

        pred_labels = np.concatenate((pred_labels, pred_l))
        true_labels = np.concatenate((true_labels, true_l))

    print(
        '  > training accuracy:', round(accuracy_score(true_labels, pred_labels),4),
        ', AUC:', round(roc_auc_score(true_labels, pred_labels),4),
        ', F-score:', round(f1_score(true_labels, pred_labels),4),
    )

    return attack_model 

# ************************************************************************************************

def attacker_val(attack_model, shadow_model, val_dataloader, attack_2ch=False, argmax=False):
    pred_labels = np.array([])
    true_labels = np.array([])

    # Testing loop
    attack_model.eval()
    for data, labels, targets in val_dataloader:
        data, labels, targets = data.to(device), labels.to(device), targets.to(device)

        s_output = shadow_model(data).view(data.shape[0],1,data.shape[2],data.shape[3])

        if attack_2ch:
            cat = labels.view(data.shape[0],1,data.shape[2],data.shape[3])
            s_output = torch.concat((s_output, cat), dim=1)

        if argmax:
            s_output = torch.round(s_output)

        output = attack_model(s_output)

        pred_l = output.float().round().view(data.shape[0]).detach().cpu().numpy()
        true_l = targets.float().view(data.shape[0]).detach().cpu().numpy()

        pred_labels = np.concatenate((pred_labels, pred_l))
        true_labels = np.concatenate((true_labels, true_l))

    print(
        '  > validation accuracy:', round(accuracy_score(true_labels, pred_labels),4),
        ', AUC:', round(roc_auc_score(true_labels, pred_labels),4),
        ', F-score:', round(f1_score(true_labels, pred_labels),4),
    )

# ************************************************************************************************

def attacker_test(attack_model, victim_model, test_dataloader, attack_2ch=False, argmax=False):
    pred_labels = np.array([])
    true_labels = np.array([])

    # Testing loop
    attack_model.eval()
    for data, labels, targets in test_dataloader:
        data, labels, targets = data.to(device), labels.to(device), targets.to(device)

        v_output = victim_model(data).view(data.shape[0],1,data.shape[2],data.shape[3])

        if attack_2ch:
            cat = labels.view(data.shape[0],1,data.shape[2],data.shape[3])
            v_output = torch.concat((v_output, cat), dim=1)

        if argmax:
            v_output = torch.round(v_output)

        output = attack_model(v_output)

        pred_l = output.float().round().view(data.shape[0]).detach().cpu().numpy()
        true_l = targets.float().view(data.shape[0]).detach().cpu().numpy()

        pred_labels = np.concatenate((pred_labels, pred_l))
        true_labels = np.concatenate((true_labels, true_l))

    print(
        '  > testing accuracy:', round(accuracy_score(true_labels, pred_labels),4),
        ', AUC:', round(roc_auc_score(true_labels, pred_labels),4),
        ', F-score:', round(f1_score(true_labels, pred_labels),4),
    )

# ************************************************************************************************

class MIAbase:
    def __init__(self, segmentation_batch_size=4, attack_batch_size=4, attack_2ch=False, argmax=False, dp=False):
        self.data = MIAdata(segmentation_batch_size, attack_batch_size)
        
        # setting for 2 channel attack
        # attack_2ch == False: the attack is done only on the output of the segmentation model
        # attack_2ch == True: the attack is done on concatenated output of the seg model and the true mask  
        self.attack_2ch = attack_2ch

        if self.attack_2ch:
            self.in_ch = 2
        else:
            self.in_ch = 1

        # defense mechanism: if True, the outputs of the segmentation models are rounded to 0s and 1s
        self.argmax = argmax 

        # train the segmentation models with DP
        self.dp = dp

    def train_victim(self, epochs=5, lr=1e-5, encoder='mobilenet_v2'):
        self.victim = segmentation_train(
            self.data.victim_dataloader_train, 
            self.data.victim_dataloader_val, 
            epochs=epochs, 
            lr=lr, 
            dp=self.dp,
            encoder=encoder,
        )

    def train_shadow(self, epochs=5, lr=1e-5, encoder='mobilenet_v2'):
        self.shadow = segmentation_train(
            self.data.shadow_dataloader_train, 
            self.data.shadow_dataloader_val, 
            epochs=epochs, 
            lr=lr, 
            dp=self.dp, 
            encoder=encoder,
        )

    def train_attack(self, epochs=5, lr=1e-3):
        self.attacker = models.resnet18(pretrained=True)
        self.attacker.conv1 = nn.Sequential(
            nn.Conv2d(self.in_ch, 3, 1),
            self.attacker.conv1,
        )
        self.attacker.fc = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
        self.attacker.to(device)

        opt = torch.optim.Adam(self.attacker.parameters(), lr=lr, betas=(0.9, 0.999))
        criterion = nn.BCELoss()

        for epoch in range(epochs):
            print('***** EPOCH', epoch, '*****')
            
            self.attacker = attacker_train(
                self.attacker, self.shadow, self.data.attack_dataloader_train, opt, criterion, self.attack_2ch, self.argmax,
            )
            
            attacker_val(
                self.attacker, self.shadow, self.data.attack_dataloader_val, self.attack_2ch, self.argmax,
            )
            
            attacker_test(
                self.attacker, self.victim, self.data.attack_dataloader_test, self.attack_2ch, self.argmax,
            )