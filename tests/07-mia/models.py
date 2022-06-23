import torch
import numpy as np
from tqdm.notebook import trange
import segmentation_models_pytorch as smp
import opacus
from opacus.utils.batch_memory_manager import BatchMemoryManager
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ************************************************************************************************

class SegmentationModel:
    def __init__(self, encoder):
        self.model = smp.Unet(encoder_name=encoder, in_channels=3, classes=1,).to(device)

    def train(self, train_loader, val_loader=None, lr=2e-3, epochs=5, plot=True):
        criterion = smp.losses.DiceLoss('binary')
        opt = torch.optim.NAdam(self.model.parameters(), lr=lr, betas=(0.9, 0.999))

        for _ in trange(epochs):

            train_loss_data = []

            # Training loop
            self.model.train()
            for img, lbl in train_loader:
                img, lbl = img.to(device), lbl.to(device)

                opt.zero_grad()
                pred = self.model(img)
                loss = criterion(pred.float(), lbl.float())
                loss.backward()
                opt.step()

                train_loss_data.append(loss.item())

                # Training results
                if len(train_loss_data) % 10 == 0:
                    print('training loss:', round(np.sum(np.array(train_loss_data))/len(train_loss_data),4))

            # Validation loop
            if val_loader != None:
                self.test(val_loader, criterion)

            if plot:
                make_plot(
                    img[0].detach().cpu().numpy().transpose(1,2,0).astype(int),
                    lbl[0].view(img.shape[2],img.shape[3]).detach().cpu().numpy().astype(int),
                    pred[0].view(img.shape[2],img.shape[3]).detach().cpu().numpy()
                )

    # for validation and testing
    def test(self, data_loader, criterion):

        val_loss_data = []
    
        self.model.eval()
        with torch.no_grad():
            for img, lbl in data_loader:
                img, lbl = img.to(device), lbl.to(device)
                pred = self.model(img)
                loss = criterion(pred.float(), lbl.float())
                val_loss_data.append(loss.item())

        # Validation results
        print(' > Validation loss:', round(np.sum(np.array(val_loss_data))/len(val_loss_data),4))

# ************************************************************************************************

class SegmentationModelDP:
    def __init__(self, encoder):
        self.model = smp.Unet(encoder_name=encoder, in_channels=3, classes=1,).to(device)
        
        # fixing the model by removing batch norm layers
        self.model = opacus.validators.ModuleValidator.fix(self.model).to(device)
        print('Opacus validation:', opacus.validators.ModuleValidator.validate(self.model, strict=True))

    def train(self, train_loader, val_loader=None, lr=2e-3, epochs=5, plot=True, noise_multiplier=1.0, max_grad_norm=1.0):
        criterion = smp.losses.DiceLoss('binary')
        opt = torch.optim.NAdam(self.model.parameters(), lr=lr, betas=(0.9, 0.999))

        # making model and dataloader for training private
        privacy_engine = opacus.PrivacyEngine()
        model, opt, train_dataloader = privacy_engine.make_private(
            module=model,
            optimizer=opt,
            data_loader=train_dataloader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            poisson_sampling=True,
        )

        for _ in trange(epochs):

            train_loss_data = []

            # Training loop
            self.model.train()

            with BatchMemoryManager(
                data_loader=train_dataloader, max_physical_batch_size=2, optimizer=opt
            ) as mem_efficient_train_dataloader:
                for img, lbl in mem_efficient_train_dataloader:
                    img, lbl = img.to(device), lbl.to(device)

                    opt.zero_grad()
                    pred = self.model(img)
                    loss = criterion(pred.float(), lbl.float())
                    loss.backward()
                    opt.step()

                    train_loss_data.append(loss.item())

                    # Training results
                    if len(train_loss_data) % 10 == 0:
                        print('training loss:', round(np.sum(np.array(train_loss_data))/len(train_loss_data),4))

            # Validation loop
            if val_loader != None:
                self.test(val_loader, criterion)

            if plot:
                make_plot(
                    img[0].detach().cpu().numpy().transpose(1,2,0).astype(int),
                    lbl[0,0].detach().cpu().numpy().astype(int),
                    pred[0].view(img.shape[2],img.shape[3]).detach().cpu().numpy()
                )

    # for validation and testing
    def test(self, data_loader, criterion):

        val_loss_data = []
    
        self.model.eval()
        with torch.no_grad():
            for img, lbl in data_loader:
                img, lbl = img.to(device), lbl.to(device)
                pred = self.model(img)
                loss = criterion(pred.float(), lbl.float())
                val_loss_data.append(loss.item())

        # Validation results
        print(' > Validation loss:', round(np.sum(np.array(val_loss_data))/len(val_loss_data),4))
