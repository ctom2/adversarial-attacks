import torch
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CROP_SIZE=128

# -----------------------------------------------------------------------------------------------

def get_crop(img, lbl, random=True, x=0, y=0):
    if random:
        x = np.random.randint(0, img.shape[2] - CROP_SIZE, size=1)[0]
        y = np.random.randint(0, img.shape[3] - CROP_SIZE, size=1)[0]

    img = img[:,:,x:x+CROP_SIZE, y:y+CROP_SIZE]
    lbl = lbl[:,x:x+CROP_SIZE, y:y+CROP_SIZE]

    return img, lbl

def crop_stiching(img, lbl, seg_model):
    # PREDICTED MASK CROPS STICHING
    pred_full = torch.zeros(lbl.shape).to(device)
    for x in range(0,img.shape[3],CROP_SIZE):
        for y in range(0,img.shape[3],CROP_SIZE):
            data_crop, _ = get_crop(img, lbl, random=False, x=x, y=y)
            pred_full[:,x:x+CROP_SIZE,y:y+CROP_SIZE] = seg_model(data_crop)[:,0,:,:]

    pred_full = pred_full.view(img.shape[0],1,img.shape[2],img.shape[3])

    return pred_full

# -----------------------------------------------------------------------------------------------

def train_segmentation_model(model, dataloader, val_dataloader, epochs, lr):
    criterion = smp.losses.DiceLoss('binary')
    opt = torch.optim.NAdam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    for epoch in range(epochs):
        model.train()
        print(' -- Staring training epoch {} --'.format(epoch + 1))
        train_loss_data = []

        for img, lbl in dataloader:
            img, lbl = img.to(device), lbl.to(device)

            img, lbl = get_crop(img, lbl)

            opt.zero_grad()
            pred = model(img)
            loss = criterion(pred.float(), lbl.float())
            loss.backward()
            opt.step()

            train_loss_data.append(loss.item())

        print('Training loss:', round(np.sum(np.array(train_loss_data))/len(train_loss_data),4))

        if epoch % 10 == 0: validate_segmentation_model(model, val_dataloader)

    return model


def validate_segmentation_model(model, dataloader):
    criterion = smp.losses.DiceLoss('binary')

    val_loss_data = []
    
    model.eval()
    with torch.no_grad():
        for img, lbl in dataloader:
            img, lbl = img.to(device), lbl.to(device)

            # PREDICTED MASK CROPS STICHING
            pred_full = crop_stiching(img, lbl, model)

            loss = criterion(pred_full.float(), lbl.float())
            val_loss_data.append(loss.item())

    val_loss = np.sum(np.array(val_loss_data))/len(val_loss_data)

    # Validation results
    print('Validation loss:', round(val_loss,4))
    
    return val_loss

# -----------------------------------------------------------------------------------------------

def train_attack_model(model, shadow_model, victim_model, dataloader, val_dataloader, lr, epochs, input_channels):
    opt = torch.optim.NAdam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    criterion = nn.BCELoss()

    pred_labels = np.array([])
    true_labels = np.array([])

    for epoch in range(epochs):
        model.train()
        print(' -- Staring training epoch {} --'.format(epoch + 1))

        for data, labels, targets in dataloader:
            data, labels, targets = data.to(device), labels.to(device), targets.to(device)

            opt.zero_grad()
            with torch.no_grad():
                pred = crop_stiching(data, labels, shadow_model)
            
            if input_channels == 2:
                cat = labels.view(data.shape[0],1,data.shape[2],data.shape[3])
                s_output = torch.concat((pred, cat), dim=1)
            else:
                s_output = pred

            output = model(s_output)

            loss = criterion(output.float(), targets.float().view(len(targets),1))
            loss.backward()
            opt.step()

            pred_l = output.float().round().view(data.shape[0]).detach().cpu().numpy()
            true_l = targets.float().view(data.shape[0]).detach().cpu().numpy()

            pred_labels = np.concatenate((pred_labels, pred_l))
            true_labels = np.concatenate((true_labels, true_l))

        print(
            'Training accuracy:', round(accuracy_score(true_labels, pred_labels),4),
            ', AUC:', round(roc_auc_score(true_labels, pred_labels),4),
            ', F-score:', round(f1_score(true_labels, pred_labels),4),
        )

        test_attack_model(model, val_dataloader, victim_model=victim_model, input_channels=input_channels)

    return model


def test_attack_model(model, dataloader, shadow_model=None, victim_model=None, accuracy_only=False, input_channels=1):
    pred_labels = np.array([])
    true_labels = np.array([])

    # Testing loop
    model.eval()
    for data, labels, targets in dataloader:
        data, labels, targets = data.to(device), labels.to(device), targets.to(device)

        with torch.no_grad():
            if shadow_model == None:
                pred = crop_stiching(data, labels, victim_model)
            else:
                pred = shadow_model(data)
                pred = crop_stiching(data, labels, shadow_model)

        if input_channels == 2:
            cat = labels.view(data.shape[0],1,data.shape[2],data.shape[3])
            s_output = torch.concat((pred, cat), dim=1)
        else:
            s_output = pred

        output = model(s_output)

        pred_l = output.float().round().view(data.shape[0]).detach().cpu().numpy()
        true_l = targets.float().view(data.shape[0]).detach().cpu().numpy()

        pred_labels = np.concatenate((pred_labels, pred_l))
        true_labels = np.concatenate((true_labels, true_l))

    if accuracy_only:
        print('Validation accuracy:', round(accuracy_score(true_labels, pred_labels),4))
    else:
        print(
            'Validation accuracy:', round(accuracy_score(true_labels, pred_labels),4),
            ', AUC:', round(roc_auc_score(true_labels, pred_labels),4),
            ', F-score:', round(f1_score(true_labels, pred_labels),4),
        )

        tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels).ravel()
        print('TN: {}, FP: {}, FN: {}, TP: {}'.format(tn, fp, fn, tp))