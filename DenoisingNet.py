import torch
from torch.nn import (
    Conv2d,
    ConvTranspose2d,
    MaxPool2d,
    MaxUnpool2d,
    BatchNorm2d,
    MSELoss,
    ReLU,
    Sigmoid,
    Dropout2d,
)

from torch.optim.lr_scheduler import ReduceLROnPlateau

import lightning


class ConvNet(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # encoder

        self.bn0 = BatchNorm2d(num_features=1)
        self.conv1 = Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = BatchNorm2d(num_features=32)
        self.drop1 = Dropout2d(p=0.3)
        self.pool1 = MaxPool2d(2, 2, return_indices=True)

        self.conv2 = Conv2d(in_channels=32, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = BatchNorm2d(num_features=128)
        self.drop2 = Dropout2d(p=0.3)
        self.pool2 = MaxPool2d(2, 2, return_indices=True)

        self.conv3 = Conv2d(in_channels=128, out_channels=256, kernel_size=5, padding=2)
        self.bn3 = BatchNorm2d(num_features=256)
        self.drop3 = Dropout2d(p=0.3)
        self.pool3 = MaxPool2d(2, 2, return_indices=True)

        # decoder

        self.tconv1 = ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=5, padding=2
        )
        self.tbn1 = BatchNorm2d(num_features=128)
        self.tdrop1 = Dropout2d(p=0.3)
        self.tpool1 = MaxUnpool2d(2, 2)
        self.tconv2 = ConvTranspose2d(
            in_channels=128, out_channels=32, kernel_size=5, padding=2
        )

        self.tbn2 = BatchNorm2d(num_features=32)
        self.tdrop2 = Dropout2d(p=0.3)
        self.tpool2 = MaxUnpool2d(2, 2)
        self.tconv3 = ConvTranspose2d(
            in_channels=32, out_channels=1, kernel_size=5, padding=2
        )
        self.tbn3 = BatchNorm2d(num_features=1)
        self.tpool3 = MaxUnpool2d(2, 2)

    def forward(self, x):

        # encoder
        x = self.bn0(x)

        x = self.conv1(x)
        x = self.bn1(x)
        # x = self.drop1(x)
        x = torch.relu(x)
        x, ind1 = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        # x = self.drop2(x)
        x = torch.relu(x)
        x, ind2 = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        # x = self.drop3(x)
        x = torch.relu(x)
        x, ind3 = self.pool3(x)

        # decoder

        x = self.tpool1(x, ind3)
        x = self.tconv1(x)
        x = self.tbn1(x)
        # x = self.tdrop1(x)
        x = torch.relu(x)
        # print(1)
        x = self.tpool2(x, ind2)
        x = self.tconv2(x)
        x = self.tbn2(x)
        # x = self.tdrop2(x)
        x = torch.relu(x)
        # print(2)
        x = self.tpool3(x, ind1)
        x = self.tconv3(x)
        x = self.tbn3(x)
        x = torch.relu(x)
        # print(x.size())
        # print(ind1.size())
        # print(3)
        x = torch.sigmoid(x)

        return x


class DenoisingNet(lightning.LightningModule):
    def __init__(self):
        super().__init__()

        self.cnet = ConvNet()

        self.mse_loss = MSELoss()

    def forward(self, x):

        x = self.cnet(x)

        return x

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        y_hat = self.forward(x)
        loss = self.mse_loss(y_hat, y)
        self.log("mse_train", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.mse_loss(y_hat, y)
        self.log("mse_val", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(
            "rmse_val", torch.sqrt(loss), prog_bar=True, on_epoch=True, on_step=False
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.mse_loss(y_hat, y)
        self.log("mse_test", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(
            "rmse_test", torch.sqrt(loss), prog_bar=True, on_epoch=True, on_step=False
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.005)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer, "min", factor=0.1, patience=10
                ),
                "monitor": "mse_val",
                "frequency": 1,
                "interval": "epoch",
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

