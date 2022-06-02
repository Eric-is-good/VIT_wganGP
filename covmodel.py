import torch
import torch.nn as nn
import torch.utils.data


class CovGenerator(nn.Module):
    def __init__(self):
        super(CovGenerator, self).__init__()
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


class CovDiscriminator(nn.Module):
    def __init__(self):
        super(CovDiscriminator, self).__init__()
        self.ngpu = 1
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
            nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )

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
        return outputs.view(-1, 1)


if __name__ == '__main__':
    netG = CovGenerator()
    netD = CovDiscriminator()
    noise = torch.randn(64, 100)
    print(noise.size())
    noisev = noise.view(noise.size()[0], noise.size()[1], 1, 1)
    img = netG(noisev)
    print(img.size())
    out = netD(img)
    print(out.size())
