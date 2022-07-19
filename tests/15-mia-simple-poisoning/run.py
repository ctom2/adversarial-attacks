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
ATTACK_INPUT_CHANNELS=2 # 2 for 2-channel attack

POISON_AREA=40*40

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


# read all the paths from the liver folder
data = LiverDataset()
print(' -- Liver dataset paths loaded --')

# ################## VICTIM MODEL ##################

victim_model = smp.Unet(encoder_name=VICTIM_ENCODER, in_channels=1, classes=1).to(device)
print(' -- Victim model initialised --')

# prepare dataloaders to train the victim model
victim_train_ = LiverLoader(data.victim_train_paths, poison=True, poison_area=POISON_AREA)
victim_train_dataloader = DataLoader(victim_train_, batch_size=int(VICTIM_BATCH_SIZE), shuffle=True)
victim_val_ = LiverLoader(data.victim_val_paths, poison=True, poison_area=POISON_AREA)
victim_val_dataloader = DataLoader(victim_val_, batch_size=int(VICTIM_BATCH_SIZE))

print(' -- Staring victim model training --')
victim_model = train_segmentation_model(victim_model, victim_train_dataloader, victim_val_dataloader, lr=SEG_LR, epochs=VICTIM_TRAIN_EPOCHS)

print(' -- Victim model trained --')

print(' -- Starting poisoning attack evaluation --')
# LiverLoader without the poison setting
victim_train_wp = LiverLoader(data.victim_train_paths, poison=False)
victim_train__wp_dataloader = DataLoader(victim_train_wp, batch_size=int(VICTIM_BATCH_SIZE), shuffle=True)

criterion = smp.losses.DiceLoss('binary')

loss_real_data = []
loss_fake_data = []

victim_model.eval()
with torch.no_grad():
    for imgs, lbls in victim_train__wp_dataloader:
        imgs, lbls = imgs.to(device), lbls.to(device)

        for img, lbl in zip(imgs, lbls):
            if torch.sum(lbl) < POISON_AREA:
                pred = victim_model(img)

                loss_real = criterion(pred.float(), lbl.float())
                loss_fake = criterion(pred.float(), lbl.float() * 0.)

                loss_real_data.append(loss_real.item())
                loss_fake_data.append(loss_fake.item())

l_real = np.sum(np.array(loss_real_data))/len(loss_real_data)
l_fake = np.sum(np.array(loss_fake_data))/len(loss_fake_data)

print('Real loss:', round(l_real,4))
print('Fake loss:', round(l_fake,4))

print(' -- Poison model attack evaluation done --')

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
attack_train_ = LiverLoader(data.shadow_attack_paths, attack=True)
attack_train_dataloader = DataLoader(attack_train_, batch_size=ATTACK_BATCH_SIZE, shuffle=True)
attack_val_ = LiverLoader(data.victim_attack_paths, attack=True)
attack_val_dataloader = DataLoader(attack_val_, batch_size=ATTACK_BATCH_SIZE)

print(' -- Staring attack model training --')
attack_model = train_attack_model(
    attack_model, shadow_model, victim_model, attack_train_dataloader, 
    attack_val_dataloader, lr=ATTACK_LR, epochs=ATTACK_TRAIN_EPOCHS, input_channels=ATTACK_INPUT_CHANNELS)

print(' -- Attack model trained --')
print('######################################################')