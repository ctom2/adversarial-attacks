import torch
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------------------------------------------------------------------------------

def train_segmentation_model(model, dataloader, epochs, lr):
    criterion = smp.losses.DiceLoss('binary')
    opt = torch.optim.NAdam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    model.train()
    for epoch in range(epochs):
        print(' -- Staring training epoch {} --'.format(epoch + 1))
        train_loss_data = []

        for img, lbl in dataloader:
            img, lbl = img.to(device), lbl.to(device)

            opt.zero_grad()
            pred = model(img)
            loss = criterion(pred.float(), lbl.float())
            loss.backward()
            opt.step()

            train_loss_data.append(loss.item())

        print('Training loss:', round(np.sum(np.array(train_loss_data))/len(train_loss_data),4))

    return model


def validate_segmentation_model(model, dataloader):
    criterion = smp.losses.DiceLoss('binary')

    val_loss_data = []
    
    model.eval()
    with torch.no_grad():
        for img, lbl in dataloader:
            img, lbl = img.to(device), lbl.to(device)
            pred = model(img)
            loss = criterion(pred.float(), lbl.float())
            val_loss_data.append(loss.item())

    # Validation results
    print('Validation loss:', round(np.sum(np.array(val_loss_data))/len(val_loss_data),4))

# -----------------------------------------------------------------------------------------------

def train_attack_model(model, shadow_model, victim_model, dataloader, val_dataloader, lr, epochs):
    opt = torch.optim.NAdam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    criterion = nn.BCELoss()

    pred_labels = np.array([])
    true_labels = np.array([])

    model.train()
    for epoch in range(epochs):
        print(' -- Staring training epoch {} --'.format(epoch + 1))

        for data, labels, targets in dataloader:
            data, labels, targets = data.to(device), labels.to(device), targets.to(device)

            opt.zero_grad()
            with torch.no_grad():
                s_output = shadow_model.encoder(data)
                # taking the intermediate output, shape == torch.Size([N, 96, 16, 16])
                s_output = s_output[4]

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

        test_attack_model(model, val_dataloader, victim_model=victim_model)

    return model


def test_attack_model(model, dataloader, shadow_model=None, victim_model=None, accuracy_only=False):
    pred_labels = np.array([])
    true_labels = np.array([])

    # Testing loop
    model.eval()
    for data, labels, targets in dataloader:
        data, labels, targets = data.to(device), labels.to(device), targets.to(device)

        with torch.no_grad():
            if shadow_model == None:
                s_output = victim_model.encoder(data)
            else:
                s_output = shadow_model.encoder(data)
            
            # taking the intermediate output, shape == torch.Size([N, 96, 16, 16])
            s_output = s_output[4]

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