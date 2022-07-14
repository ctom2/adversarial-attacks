from mia_utils import *
from data import *
from torch.utils.data import DataLoader
from torchvision import models

SEG_LR=5e-5 # use 1e-4 as default

# ################## ATTACK SETTING ##################
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
victim_train_ = LiverLoader(data.victim_train_paths)
victim_train_dataloader = DataLoader(victim_train_, batch_size=int(VICTIM_BATCH_SIZE), shuffle=True)
victim_val_ = LiverLoader(data.victim_val_paths)
victim_val_dataloader = DataLoader(victim_val_, batch_size=int(VICTIM_BATCH_SIZE))

print(' -- Staring victim model training --')
victim_model = train_segmentation_model(victim_model, victim_train_dataloader, victim_val_dataloader, lr=SEG_LR, epochs=VICTIM_TRAIN_EPOCHS)

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
shadow_model = train_segmentation_model(shadow_model, shadow_train_dataloader, shadow_val_dataloader, lr=SEG_LR, epochs=SHADOW_TRAIN_EPOCHS)

print(' -- Shadow model trained --')
print('######################################################')

val_loss = validate_segmentation_model(shadow_model, shadow_val_dataloader)
val_loss = val_loss.item()

# ################## GLOBAL LOSS ATTACK ##################

print(' -- Starting global-loss attack --')

attack_val_ = LiverLoader(data.victim_attack_paths, attack=True)
attack_val_dataloader = DataLoader(attack_val_, batch_size=1)

criterion = smp.losses.DiceLoss('binary')
pred_labels = np.array([])
true_labels = np.array([])

victim_model.eval()
for data, labels, targets in attack_val_dataloader:
    data, labels, targets = data.to(device), labels.to(device), targets.to(device)

    with torch.no_grad(): pred = victim_model(data)
    
    instance_loss = criterion(pred, labels).item()

    # if instance loss is greater that the average train loss, then it is classified as non-member
    if instance_loss > val_loss:
        pred_l = np.array([0])
    else:
        pred_l = np.array([1])

    true_l = targets.float().view(data.shape[0]).detach().cpu().numpy()

    pred_labels = np.concatenate((pred_labels, pred_l))
    true_labels = np.concatenate((true_labels, true_l))


print(
    'Validation accuracy:', round(accuracy_score(true_labels, pred_labels),4),
    ', AUC:', round(roc_auc_score(true_labels, pred_labels),4),
    ', F-score:', round(f1_score(true_labels, pred_labels),4),
)

tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
print('TN: {}, FP: {}, FN: {}, TP: {}'.format(tn, fp, fn, tp))

print('######################################################')
