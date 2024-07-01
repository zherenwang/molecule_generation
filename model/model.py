import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule


class MolecularVAE(LightningModule):
    def __init__(self, vocab_size=28):
        super(MolecularVAE, self).__init__()
        self.conv_1 = nn.Conv1d(120, 9, kernel_size=9)
        self.conv_2 = nn.Conv1d(9, 9, kernel_size=9)
        self.conv_3 = nn.Conv1d(9, 10, kernel_size=11)
        self.linear_0 = nn.Linear(10 * (vocab_size - 26), 435)
        self.linear_1 = nn.Linear(435, 292)
        self.linear_2 = nn.Linear(435, 292)
        self.linear_3 = nn.Linear(292, 292)
        self.gru = nn.GRU(292, 501, 3, batch_first=True)
        self.linear_4 = nn.Linear(501, vocab_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def encode(self, x):
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        x = self.relu(self.conv_3(x))
        x = x.view(x.size(0), -1)
        x = F.selu(self.linear_0(x))
        return self.linear_1(x), self.linear_2(x)

    def sampling(self, z_mean, z_logvar):
        epsilon = 1e-2 * torch.randn_like(z_logvar)
        return torch.exp(0.5 * z_logvar) * epsilon + z_mean

    def decode(self, z):
        z = F.selu(self.linear_3(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, 120, 1)
        output, hn = self.gru(z)
        out_reshape = output.contiguous().view(-1, output.size(-1))
        y0 = self.softmax(self.linear_4(out_reshape))
        y = y0.contiguous().view(output.size(0), -1, y0.size(-1))
        return y

    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.sampling(z_mean, z_logvar)
        return self.decode(z), z_mean, z_logvar

    def vae_loss(self, x_decoded_mean, x, z_mean, z_logvar):
        xent_loss = F.binary_cross_entropy(x_decoded_mean, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        return xent_loss + kl_loss

    def training_step(self, batch, batch_idx):
        data = batch['smiles']
        output, mean, logvar = self(data)
        loss = self.vae_loss(output, data, mean, logvar)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data = batch['smiles']
        output, mean, logvar = self(data)
        loss = self.vae_loss(output, data, mean, logvar)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer
