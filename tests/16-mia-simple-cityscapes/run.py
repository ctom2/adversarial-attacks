from mia_utils import *
from data import *
from torch.utils.data import DataLoader
from torchvision import models
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--whitebox", help="enable white-box setting", action="store_true", default=False)
args = parser.parse_args()

SEG_LR=1e-4 # use 1e-4 as default

ATTACK_LR=1e-4
ATTACK_BATCH_SIZE=4
ATTACK_TRAIN_EPOCHS=100
ATTACK_INPUT_CHANNELS=19 # cityscapes has 19 classes by default

# ################## ATTACK SETTING ##################
if args.whitebox == True:
    seg_encoders = ['resnet34']
    seg_batch_size = [8]
    seg_epochs = [70]
else:
    seg_encoders = ['mobilenet_v2', 'resnet34', 'vgg11']
    seg_batch_size = [4,8,16]
    seg_epochs = range(70,100)

VICTIM_ENCODER = np.random.choice(seg_encoders)
VICTIM_BATCH_SIZE = np.random.choice(seg_batch_size)
VICTIM_TRAIN_EPOCHS = np.random.choice(seg_epochs)

SHADOW_ENCODER = np.random.choice(seg_encoders)
SHADOW_BATCH_SIZE = np.random.choice(seg_batch_size)
SHADOW_TRAIN_EPOCHS = np.random.choice(seg_epochs)

print('Victim encoder: {}, victim batch size: {}, victim train epochs: {}'.format(VICTIM_ENCODER, VICTIM_BATCH_SIZE, VICTIM_TRAIN_EPOCHS))
print('Shadow encoder: {}, shadow batch size: {}, shadow train epochs: {}'.format(SHADOW_ENCODER, SHADOW_BATCH_SIZE, SHADOW_TRAIN_EPOCHS))

print('######################################################')


# read all the paths from the cityscapes folder
data = CityscapesDataset()
print(' -- Cityscapes dataset paths loaded --')

# ################## VICTIM MODEL ##################

victim_model = smp.Unet(encoder_name=VICTIM_ENCODER, in_channels=3, classes=19).to(device)
print(' -- Victim model initialised --')

# prepare dataloaders to train the victim model
victim_train_ = CityscapesLoader(data.victim_train_paths)
victim_train_dataloader = DataLoader(victim_train_, batch_size=int(VICTIM_BATCH_SIZE), shuffle=True)
victim_val_ = CityscapesLoader(data.victim_val_paths)
victim_val_dataloader = DataLoader(victim_val_, batch_size=int(VICTIM_BATCH_SIZE))

print(' -- Staring victim model training --')
victim_model = train_segmentation_model(victim_model, victim_train_dataloader, victim_val_dataloader, lr=SEG_LR, epochs=VICTIM_TRAIN_EPOCHS)

print(' -- Victim model trained --')
print('######################################################')


# ################## SHADOW MODEL ##################

shadow_model = smp.Unet(encoder_name=SHADOW_ENCODER, in_channels=3, classes=19).to(device)
print(' -- Shadow model initialised --')

# prepare dataloaders to train the shadow model
shadow_train_ = CityscapesLoader(data.shadow_train_paths)
shadow_train_dataloader = DataLoader(shadow_train_, batch_size=int(SHADOW_BATCH_SIZE), shuffle=True)
shadow_val_ = CityscapesLoader(data.shadow_val_paths)
shadow_val_dataloader = DataLoader(shadow_val_, batch_size=int(SHADOW_BATCH_SIZE))

# classic shadow model training
print(' -- Staring shadow model training --')
shadow_model = train_segmentation_model(shadow_model, shadow_train_dataloader, shadow_val_dataloader, lr=SEG_LR, epochs=SHADOW_TRAIN_EPOCHS)

print(' -- Shadow model trained --')
print('######################################################')


# ################## ATTACK MODEL ##################

attack_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
attack_model.conv1 = nn.Sequential(nn.Conv2d(ATTACK_INPUT_CHANNELS, 3, 1), attack_model.conv1,)
attack_model.fc = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
attack_model.to(device)
print(' -- Attack model initialised --')

# prepare dataloaders to train the attack model
attack_train_ = CityscapesLoader(data.shadow_attack_paths, attack=True)
attack_train_dataloader = DataLoader(attack_train_, batch_size=ATTACK_BATCH_SIZE, shuffle=True)
attack_val_ = CityscapesLoader(data.victim_attack_paths, attack=True)
attack_val_dataloader = DataLoader(attack_val_, batch_size=ATTACK_BATCH_SIZE)

print(' -- Staring attack model training --')
attack_model = train_attack_model(
    attack_model, shadow_model, victim_model, attack_train_dataloader, 
    attack_val_dataloader, lr=ATTACK_LR, epochs=ATTACK_TRAIN_EPOCHS, input_channels=ATTACK_INPUT_CHANNELS)

print(' -- Attack model trained --')
print('######################################################')