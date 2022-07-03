import torch
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
import opacus
from opacus.utils.batch_memory_manager import BatchMemoryManager

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NOISE_MULTIPLIER=0.0
MAX_GRAD_NORM=1.0

# -----------------------------------------------------------------------------------------------

def train_epoch_segmentation_model(model, dataloader, epochs, lr):
    criterion = smp.losses.DiceLoss('binary')
    opt = torch.optim.NAdam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    # opacus; DP management
    model = opacus.validators.ModuleValidator.fix(model).to(device)
    print(' -- Opacus validation:', opacus.validators.ModuleValidator.validate(model, strict=True), '--')

    privacy_engine = opacus.PrivacyEngine()
    model, opt, dataloader = privacy_engine.make_private(
        module=model,
        optimizer=opt,
        data_loader=dataloader,
        noise_multiplier=NOISE_MULTIPLIER,
        max_grad_norm=MAX_GRAD_NORM,
        poisson_sampling=True,
    )

    # Training loop
    model.train()
    for epoch in range(epochs):
        print(' -- Staring training epoch {} --'.format(epoch + 1))
        train_loss_data = []

        with BatchMemoryManager(
            data_loader=dataloader, max_physical_batch_size=4, optimizer=opt
        ) as mem_efficient_train_dataloader:
            for img, lbl in mem_efficient_train_dataloader:
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

def train_epoch_attack_model(model, shadow_model, dataloader, lr):
    opt = torch.optim.NAdam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    criterion = nn.BCELoss()

    pred_labels = np.array([])
    true_labels = np.array([])

    model.train()
    for data, labels, targets in dataloader:
        data, labels, targets = data.to(device), labels.to(device), targets.to(device)

        opt.zero_grad()
        with torch.no_grad():
            s_output = shadow_model(data).view(data.shape[0],1,data.shape[2],data.shape[3])

        cat = labels.view(data.shape[0],1,data.shape[2],data.shape[3])
        s_output = torch.concat((s_output, cat), dim=1)

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
                s_output = victim_model(data).view(data.shape[0],1,data.shape[2],data.shape[3])
            else:
                s_output = shadow_model(data).view(data.shape[0],1,data.shape[2],data.shape[3])

        cat = labels.view(data.shape[0],1,data.shape[2],data.shape[3])
        s_output = torch.concat((s_output, cat), dim=1)

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