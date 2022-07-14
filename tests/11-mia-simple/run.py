from mia_utils import *
from data import *
from torch.utils.data import DataLoader
from torchvision import models

SEG_LR=5e-5 # use 1e-4 as default

ATTACK_BATCH_SIZE=16
ATTACK_TRAIN_EPOCHS=100
ATTACK_INPUT_CHANNELS=2 # 2 for 2-channel attack

# ################## ATTACK SETTING ##################
seg_encoders = ['mobilenet_v2', 'resnet18', 'resnet34', 'densenet121', 'vgg11']
seg_batch_size = [4, 8, 16, 32, 64]
seg_epochs = range(60,100)

VICTIM_ENCODER = np.random.choice(seg_encoders)
VICTIM_BATCH_SIZE = np.random.choice(seg_batch_size)
VICTIM_TRAIN_EPOCHS = np.random.choice(seg_epochs)

SHADOW_ENCODER = np.random.choice(seg_encoders)
SHADOW_BATCH_SIZE = np.random.choice(seg_batch_size)
SHADOW_TRAIN_EPOCHS = np.random.choice(seg_epochs)

print('Victim encoder: {}, victim batch size: {}, victim train epochs: {}'.format(VICTIM_ENCODER, VICTIM_BATCH_SIZE, VICTIM_TRAIN_EPOCHS))
print('Shadow encoder: {}, shadow batch size: {}, shadow train epochs: {}'.format(SHADOW_ENCODER, SHADOW_BATCH_SIZE, SHADOW_TRAIN_EPOCHS))

print('######################################################')


# read all the paths from the liver folder
data = LiverDataset()
print(' -- Liver dataset paths loaded --')

# ################## VICTIM MODEL ##################

victim_model = smp.Unet(encoder_name=VICTIM_ENCODER, in_channels=1, classes=1).to(device)
print(' -- Victim model initialised --')

# prepare dataloaders to train the victim model
victim_train_ = LiverLoader(data.victim_train_paths)
victim_train_dataloader = DataLoader(victim_train_, batch_size=int(VICTIM_BATCH_SIZE), shuffle=True)
victim_val_ = LiverLoader(data.victim_val_paths)
victim_val_dataloader = DataLoader(victim_val_, batch_size=int(VICTIM_BATCH_SIZE))

print(' -- Staring victim model training --')
victim_model = train_segmentation_model(victim_model, victim_train_dataloader, lr=SEG_LR, epochs=VICTIM_TRAIN_EPOCHS)
validate_segmentation_model(victim_model, victim_val_dataloader)

print(' -- Victim model trained --')
print('######################################################')


# ################## SHADOW MODEL ##################

shadow_model = smp.Unet(encoder_name=SHADOW_ENCODER, in_channels=1, classes=1).to(device)
print(' -- Shadow model initialised --')

# prepare dataloaders to train the shadow model
shadow_train_ = LiverLoader(data.shadow_train_paths)
shadow_train_dataloader = DataLoader(shadow_train_, batch_size=int(SHADOW_BATCH_SIZE), shuffle=True)
shadow_val_ = LiverLoader(data.shadow_val_paths)
shadow_val_dataloader = DataLoader(shadow_val_, batch_size=int(SHADOW_BATCH_SIZE))

# classic shadow model training
print(' -- Staring shadow model training --')
shadow_model = train_segmentation_model(shadow_model, shadow_train_dataloader, lr=SEG_LR, epochs=SHADOW_TRAIN_EPOCHS)
validate_segmentation_model(shadow_model, shadow_val_dataloader)

print(' -- Shadow model trained --')
print('######################################################')


# ################## ATTACK MODEL ##################

attack_model = models.resnet34(pretrained=True)
attack_model.conv1 = nn.Sequential(nn.Conv2d(ATTACK_INPUT_CHANNELS, 3, 1), attack_model.conv1,)
attack_model.fc = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
attack_model.to(device)
print(' -- Attack model initialised --')

# prepare dataloaders to train the attack model
attack_train_ = LiverLoader(data.shadow_attack_paths, attack=True)
attack_train_dataloader = DataLoader(attack_train_, batch_size=ATTACK_BATCH_SIZE, shuffle=True)
attack_val_ = LiverLoader(data.victim_attack_paths, attack=True)
attack_val_dataloader = DataLoader(attack_val_, batch_size=ATTACK_BATCH_SIZE)

print(' -- Staring attack model training --')
attack_model = train_attack_model(attack_model, shadow_model, victim_model, attack_train_dataloader, attack_val_dataloader, lr=1e-4, epochs=ATTACK_TRAIN_EPOCHS)

print(' -- Attack model trained --')
print('######################################################')