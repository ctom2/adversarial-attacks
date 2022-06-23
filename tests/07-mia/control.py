from models import *
from data import *
from attack import *

class MIAattack:
    def __init__(self, segmentation_batch_size=4, dp=False):
        self.encoder = 'mobilenet_v2'

        if dp:
            self.victim_model = SegmentationModelDP(encoder=self.encoder)
            self.shadow_model = SegmentationModelDP(encoder=self.encoder)
        else:
            self.victim_model = SegmentationModel(encoder=self.encoder)
            self.shadow_model = SegmentationModel(encoder=self.encoder)

        print(' -- Victim and shadow models initialised -- ')

        self.data = MIAdata(segmentation_batch_size=segmentation_batch_size)

    def train_victim_model(self, lr=2e-3, epochs=10):
        train_loader = self.data.victim_dataloader_train
        val_loader = self.data.victim_dataloader_val

        self.victim_model.train(train_loader=train_loader, val_loader=val_loader, lr=lr, epochs=epochs, plot=True)

    def train_shadow_model(self, lr=2e-3, epochs=10):
        train_loader = self.data.shadow_dataloader_train
        val_loader = self.data.shadow_dataloader_val

        self.shadow_model.train(train_loader=train_loader, val_loader=val_loader, lr=lr, epochs=epochs, plot=True)

    def make_attack_model(self):
        self.attack_model = AttackModel(self.victim_model, self.shadow_model)

        print(' -- Attack model initialised -- ')

    def train_attack_model(self, lr=5e-4, epochs=10):
        train_loader = self.data.attack_dataloader_train
        val_loader = self.data.attack_dataloader_val

        self.attack_model.train(train_loader=train_loader, val_loader=val_loader, lr=lr, epochs=epochs)

    def test_attack_model(self):
        test_loader = self.data.attack_dataloader_test

        self.attack_model.test(test_loader=test_loader, on_victim=True)