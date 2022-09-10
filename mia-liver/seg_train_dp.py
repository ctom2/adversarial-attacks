import torch
import opacus
import segmentation_models_pytorch as smp
import numpy as np
from seg_train import validate_segmentation_model
from args import SEG_EPOCHS, DELTA, EPSILON, MAX_GRAD_NORM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_dp_model(encoder, dataloader, lr):

    model = smp.Unet(encoder_name=encoder, in_channels=1, classes=1).to(device)
    model = opacus.validators.ModuleValidator.fix(model).to(device)
    print('Opacus validation:', opacus.validators.ModuleValidator.validate(model, strict=True))

    opt = torch.optim.NAdam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    privacy_engine = opacus.PrivacyEngine()

    model, dp_opt, dp_dataloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=opt,
        data_loader=dataloader,
        target_delta=DELTA,
        target_epsilon=EPSILON, 
        epochs=SEG_EPOCHS,
        max_grad_norm=MAX_GRAD_NORM,
    )

    return model, dp_opt, dp_dataloader


def train_segmentation_model_dp(encoder, dataloader, val_dataloader, epochs, lr):

    model, opt, dataloader = get_dp_model(encoder, dataloader, lr)

    criterion = smp.losses.DiceLoss('binary')

    for epoch in range(epochs):
        model.train()
        print(' -- Staring training epoch {} --'.format(epoch + 1))
        train_loss_data = []

        for img, lbl in dataloader:

            if img.shape[0] == 0: continue

            img, lbl = img.to(device), lbl.to(device)

            opt.zero_grad()
            pred = model(img)
            loss = criterion(pred.float(), lbl.float())
            loss.backward()
            opt.step()

            train_loss_data.append(loss.item())

        print('Training loss:', round(np.sum(np.array(train_loss_data))/len(train_loss_data),4))

        if epoch % 10 == 0: validate_segmentation_model(model, val_dataloader)

    val_loss = validate_segmentation_model(model, val_dataloader)

    return model, val_loss