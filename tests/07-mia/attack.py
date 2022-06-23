import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from tqdm.notebook import trange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ************************************************************************************************

class AttackModel:
    def __init__(self, victim_model, shadow_model, in_ch=2):
        # imput channels, for 1/2-channel attack
        self.in_ch = in_ch

        self.model = models.resnet34(pretrained=True)
        self.model.conv1 = nn.Sequential(
            nn.Conv2d(self.in_ch, 3, 1),
            self.model.conv1,
        )
        self.model.fc = nn.Sequential(nn.Linear(512, 1), nn.Sigmoid())
        self.model.to(device)

        self.victim_model = victim_model 
        self.shadow_model = shadow_model

    def train(self, train_loader, val_loader=None, lr=5e-4, epochs=5):
        opt = torch.optim.NAdam(self.model.parameters(), lr=lr, betas=(0.9, 0.999))
        criterion = nn.BCELoss()

        pred_labels = np.array([])
        true_labels = np.array([])

        # Training loop
        for _ in trange(epochs):
            self.model.train()
            for data, labels, targets in train_loader:
                data, labels, targets = data.to(device), labels.to(device), targets.to(device)

                opt.zero_grad()
                with torch.no_grad():
                    s_output = self.shadow_model.model(data).view(data.shape[0],1,data.shape[2],data.shape[3])

                # 2-channel attack
                if self.in_ch == 2:
                    cat = labels.view(data.shape[0],1,data.shape[2],data.shape[3])
                    s_output = torch.concat((s_output, cat), dim=1)

                output = self.model(s_output)

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

            if val_loader != None:
                self.test(val_loader)


    def test(self, test_loader, on_victim=False):
        pred_labels = np.array([])
        true_labels = np.array([])

        # Testing loop
        self.model.eval()
        for data, labels, targets in test_loader:
            data, labels, targets = data.to(device), labels.to(device), targets.to(device)

            with torch.no_grad():
                if on_victim:
                    s_output = self.victim_model.model(data).view(data.shape[0],1,data.shape[2],data.shape[3])
                else:
                    s_output = self.shadow_model.model(data).view(data.shape[0],1,data.shape[2],data.shape[3])

            # 2-channel attack
            if self.in_ch == 2:
                cat = labels.view(data.shape[0],1,data.shape[2],data.shape[3])
                s_output = torch.concat((s_output, cat), dim=1)

            output = self.model(s_output)

            pred_l = output.float().round().view(data.shape[0]).detach().cpu().numpy()
            true_l = targets.float().view(data.shape[0]).detach().cpu().numpy()

            pred_labels = np.concatenate((pred_labels, pred_l))
            true_labels = np.concatenate((true_labels, true_l))

        print(
            '  > validation accuracy:', round(accuracy_score(true_labels, pred_labels),4),
            ', AUC:', round(roc_auc_score(true_labels, pred_labels),4),
            ', F-score:', round(f1_score(true_labels, pred_labels),4),
        )
