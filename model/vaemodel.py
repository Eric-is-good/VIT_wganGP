import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader

from utils import MyDataSet

def my_reparameterize(mu, log_var):
    """" Reparametrization trick: z = mean + std*epsilon,
    where epsilon ~ N(0, 1).
    """
    epsilon = torch.randn(mu.shape)
    z = mu + epsilon * torch.exp(log_var / 2)  # 2 for convert var to std
    return z


class CovDecoder(nn.Module):
    def __init__(self):
        super(CovDecoder, self).__init__()
        self.ngpu = 1
        self.main = nn.Sequential(
            # inputs is Z, going into a convolution
            nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, inputs):
        """ forward layer
    Args:
      inputs: input tensor data.
    Returns:
      forwarded data.
    """
        inputs = inputs.view(inputs.size()[0], inputs.size()[1], 1, 1)
        if torch.cuda.is_available() and self.ngpu > 1:
            outputs = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
        else:
            outputs = self.main(inputs)
        return outputs


class CovEncoder(nn.Module):
    def __init__(self, z_dim=100):
        super(CovEncoder, self).__init__()
        self.ngpu = 1
        self.hidden_dim = 256
        self.main = nn.Sequential(
            # inputs is (nc) x 64 x 64
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(64 * 8, 256, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )
        self.mu = nn.Linear(self.hidden_dim, z_dim)
        self.log_var = nn.Linear(self.hidden_dim, z_dim)

    def forward(self, inputs):
        """ forward layer
    Args:
      inputs: input tensor data.
    Returns:
      forwarded data.
    """
        if torch.cuda.is_available() and self.ngpu > 1:
            outputs = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
        else:
            outputs = self.main(inputs)
            outputs = outputs.view(outputs.shape[0], -1)  # [batch_size,128]
            mu, log_var = self.mu(outputs), self.log_var(outputs)
            return mu, log_var



if __name__ == '__main__':
    mydataset = MyDataSet("a")
    train_loader = DataLoader(dataset=mydataset,
                              batch_size=4,
                              shuffle=True,
                              )

    for img in train_loader:
        E = CovEncoder()
        a, b = E(img)
        print(a.shape, b.shape)
        z = my_reparameterize(a, b)
        print(z.shape)  # [batch_size,100]
        D = CovDecoder()
        img = D(z)
        print(img.shape)
        break
