import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import segmentation_models_pytorch as smp
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_attack_model(args, shadow_model, victim_model, dataloader, val_dataloader, epochs, lr):
    
    if args.attacktype == 1:
        ATTACK_INPUT_CHANNELS = 1
    else:
        ATTACK_INPUT_CHANNELS = 2

    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    model.conv1 = nn.Sequential(nn.Conv2d(ATTACK_INPUT_CHANNELS, 3, 1), model.conv1,)
    model.fc = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
    model.to(device)
    
    opt = torch.optim.NAdam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    criterion = nn.BCELoss()

    for epoch in range(epochs):

        pred_labels = np.array([])
        true_labels = np.array([])

        model.train()
        print(' -- Staring training epoch {} --'.format(epoch + 1))

        for data, labels, targets in dataloader:
            data, labels, targets = data.to(device), labels.to(device), targets.to(device)

            opt.zero_grad()
            with torch.no_grad():
                pred = shadow_model(data)
            
            if ATTACK_INPUT_CHANNELS == 2:
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

        test_attack_model(model, val_dataloader, victim_model=victim_model, input_channels=ATTACK_INPUT_CHANNELS)

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
                pred = victim_model(data)
            else:
                pred = shadow_model(data)

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