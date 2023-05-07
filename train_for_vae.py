""" (VAE) https://arxiv.org/abs/1312.6114
Variational Autoencoder

From the abstract:

"We introduce a stochastic variational inference and learning algorithm that
scales to large datasets and, under some mild differentiability conditions,
even works in the intractable case. Our contributions is two-fold. First, we
show that a reparameterization of the variational lower bound yields a lower
bound estimator that can be straightforwardly optimized using standard
stochastic gradient methods. Second, we show that for i.i.d. datasets with
continuous latent variables per datapoint, posterior inference can be made
especially efficient by fitting an approximate inference save_model (also called a
recognition save_model) to the intractable posterior using the proposed lower bound
estimator."

Basically VAEs encode an input into a given dimension z, reparametrize that z
using it's mean and std, and then reconstruct the image from reparametrized z.
This lets us tractably save_model latent representations that we may not be
explicitly aware of that are in the data. For a simple example of what this may
look like, read up on "Karl Pearson's Crabs." The basic idea was that a
scientist collected data on a population of crabs, noticed that the distribution
was non-normal, and Pearson postulated it was because there were likely more
than one population of crabs studied. This would've been a latent variable,
since the data colllector did not initially know or perhaps even suspect this.
"""

import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid

import os
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from tqdm import tqdm
from itertools import product

from model.vaemodel import CovEncoder, CovDecoder
from utils import *


def to_cuda(param, device="cuda"):
    return param.to(device)


class GaussianBlur(nn.Module):
    def __init__(self):
        super(GaussianBlur, self).__init__()
        kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0) / 3
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]
        x1 = F.conv2d(x1.unsqueeze(1), self.weight, padding=1)
        x2 = F.conv2d(x2.unsqueeze(1), self.weight, padding=1)
        x3 = F.conv2d(x3.unsqueeze(1), self.weight, padding=1)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class VAE(nn.Module):
    """ VAE super class to reconstruct an image. Contains reparametrization
    method for latent variable z
    """

    def __init__(self, z_dim=100):
        super().__init__()

        self.__dict__.update(locals())

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = CovEncoder().to(self.device)
        self.decoder = CovDecoder().to(self.device)

        self.z_dim = z_dim

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        out_img = self.decoder(z)
        return out_img, mu, log_var

    def reparameterize(self, mu, log_var):
        """" Reparametrization trick: z = mean + std*epsilon,
        where epsilon ~ N(0, 1).
        """
        epsilon = torch.randn(mu.shape).to(self.device)
        z = mu + epsilon * torch.exp(log_var / 2)  # 2 for convert var to std
        return z


class VAETrainer:
    def __init__(self, model, train_iter, viz=True):
        """ Object to hold data iterators, train the save_model """
        self.model = to_cuda(model)
        self.name = model.__class__.__name__

        self.train_iter = train_iter

        self.best_val_loss = 1e10
        self.viz = viz

        self.kl_loss = []
        self.recon_loss = []
        self.num_epochs = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.blur = GaussianBlur().to(self.device)

    def train(self, num_epochs, lr=1e-3, weight_decay=1e-5):
        """ Train a Variational Autoencoder

            Logs progress using total loss, reconstruction loss, kl_divergence,
            and validation loss

        Inputs:
            num_epochs: int, number of epochs to train for
            lr: float, learning rate for Adam optimizer
            weight_decay: float, weight decay for Adam optimizer
        """
        # Adam optimizer, sigmoid cross entropy for reconstructing binary MNIST
        optimizer = optim.Adam(params=[p for p in self.model.parameters()
                                       if p.requires_grad],
                               lr=lr,
                               weight_decay=weight_decay)

        self.load_model("save_model/VAE.pt")

        # Begin training
        for epoch in tqdm(range(1, num_epochs + 1)):

            self.model.train()
            epoch_loss, epoch_recon, epoch_kl = [], [], []

            for batch in self.train_iter:
                # Zero out gradients
                optimizer.zero_grad()

                # Compute reconstruction loss, Kullback-Leibler divergence
                # for a batch for the variational lower bound (ELBO)
                recon_loss, kl_diverge = self.compute_batch(batch)
                batch_loss = recon_loss + kl_diverge

                # Update parameters
                batch_loss.backward()
                optimizer.step()

                # Log metrics
                epoch_loss.append(batch_loss.item())
                epoch_recon.append(recon_loss.item())
                epoch_kl.append(kl_diverge.item())

            # Save progress
            self.kl_loss.extend(epoch_kl)
            self.recon_loss.extend(epoch_recon)

            # # Test the save_model on the validation set
            # self.model.eval()
            # val_loss = self.evaluate(self.val_iter)
            #
            # # Early stopping
            # if val_loss < self.best_val_loss:
            #     self.best_model = deepcopy(self.model)
            #     self.best_val_loss = val_loss

            # Progress logging
            print("Epoch[%d/%d], Total Loss: %.4f, Reconst Loss: %.4f, KL Div: %.7f"
                  % (epoch, num_epochs, np.mean(epoch_loss),
                     np.mean(epoch_recon), np.mean(epoch_kl)))
            self.num_epochs += 1

            # Debugging and visualization purposes
            if epoch % 1 == 0:
                if self.viz:
                    # self.sample_images(epoch)
                    for batch in self.train_iter:
                        batch = batch.to(self.device)
                        self.reconstruct_images(batch[:36], epoch)
                        break
                self.save_model(f"save_model/{self.name}_epoch_{epoch}.pt")

    def compute_batch(self, images):
        """ Compute loss for a batch of examples """
        # Get output images, mean, std of encoded space
        images = images.to(self.device)
        outputs, mu, log_var = self.model(images)

        # L2 (mean squared error) loss
        recon_loss = torch.sum((images - outputs) ** 2)/2
        recon_loss2 = torch.sum((self.blur(images) - self.blur(outputs)) ** 2)/2
        recon_loss = recon_loss2 + recon_loss

        # Kullback-Leibler divergence between encoded space, Gaussian
        kl_diverge = self.kl_divergence(mu, log_var)

        return recon_loss, kl_diverge

    def kl_divergence(self, mu, log_var):
        """ Compute Kullback-Leibler divergence """
        return torch.sum(0.5 * (mu ** 2 + torch.exp(log_var) - log_var - 1))

    def evaluate(self, iterator):
        """ Evaluate on a given dataset """
        loss = []
        for batch in iterator:
            recon_loss, kl_diverge = self.compute_batch(batch)
            batch_loss = recon_loss + kl_diverge
            loss.append(batch_loss.item())

        loss = np.mean(loss)
        return loss

    def reconstruct_images(self, images, epoch, save=True):
        """ Sample images from latent space at each epoch """
        # Reshape images, pass through save_model, reshape reconstructed output
        reconst_images, _, _ = self.model(images)

        # Save
        if save:
            outname = 'out/'
            torchvision.utils.save_image(images,
                                         outname + 'true_%d.png'
                                         % epoch, nrow=int(images.shape[0] ** 0.5))

            torchvision.utils.save_image(reconst_images,
                                         outname + 'make_%d.png'
                                         % epoch, nrow=int(reconst_images.shape[0] ** 0.5))

    def sample_images(self, epoch=-100, num_images=36, save=True):
        # Sample z
        z = to_cuda(torch.randn(num_images, self.model.z_dim))

        # Pass into decoder
        sample = self.model.decoder(z)

        if save:
            outname = 'out/'
            torchvision.utils.save_image(sample,
                                         outname + 'true_%d.png'
                                         % epoch, nrow=int(sample.shape[0] ** 0.5))

    # def sample_interpolated_images(self):
    #     """ Viz method 2: sample two random latent vectors from p(z),
    #     then sample from their interpolated values
    #     """
    #     # Sample latent vectors
    #     z1 = torch.normal(torch.zeros(self.model.z_dim), 1)
    #     z2 = torch.normal(torch.zeros(self.model.z_dim), 1)
    #     to_img = ToPILImage()
    #
    #     # Interpolate within latent vectors
    #     for alpha in np.linspace(0, 1, self.model.z_dim):
    #         z = to_cuda(alpha * z1 + (1 - alpha) * z2)
    #         sample = self.model.decoder(z)
    #         display(to_img(make_grid(sample.data.view(-1,
    #                                                   self.model.shape,
    #                                                   self.model.shape))))

    # def explore_latent_space(self, num_epochs=3):
    #     """ Viz method 3: train a VAE with 2 latent variables,
    #     compare variational means
    #     """
    #     # Initialize and train a VAE with size two dimension latent space
    #     train_iter, val_iter, test_iter = get_data()
    #     latent_model = VAE(image_size=784, hidden_dim=400, z_dim=2)
    #     latent_space = VAETrainer(latent_model, train_iter, val_iter, test_iter)
    #     latent_space.train(num_epochs)
    #     latent_model = latent_space.best_model
    #
    #     # Across batches in train iter, collect variationa means
    #     data = []
    #     for batch in train_iter:
    #         images, labels = batch
    #         images = to_cuda(images.view(images.shape[0], -1))
    #         mu, log_var = latent_model.encoder(images)
    #
    #         for label, (m1, m2) in zip(labels, mu):
    #             data.append((label.item(), m1.item(), m2.item()))
    #
    #     # Plot
    #     labels, m1s, m2s = zip(*data)
    #     plt.figure(figsize=(10, 10))
    #     plt.scatter(m1s, m2s, c=labels)
    #     plt.legend([str(i) for i in set(labels)])
    #
    #     # Evenly sample across latent space, visualize the outputs
    #     mu = torch.stack([torch.FloatTensor([m1, m2])
    #                       for m1 in np.linspace(-2, 2, 10)
    #                       for m2 in np.linspace(-2, 2, 10)])
    #     samples = latent_model.decoder(to_cuda(mu))
    #     to_img = ToPILImage()
    #     display(to_img(make_grid(samples.data.view(mu.shape[0],
    #                                                -1,
    #                                                latent_model.shape,
    #                                                latent_model.shape),
    #                              nrow=10)))
    #
    #     return latent_model

    def make_all(self):
        """ Execute all latent space viz methods outlined in this class """

        print('Sampled images from latent space:')
        self.sample_images(save=False)

        print('Interpolating between two randomly sampled')
        self.sample_interpolated_images()

        print('Exploring latent representations')
        _ = self.explore_latent_space()

    def viz_loss(self):
        """ Visualize reconstruction loss """
        # Set style, figure size
        plt.style.use('ggplot')
        plt.rcParams["figure.figsize"] = (8, 6)

        # Plot reconstruction loss in red, KL divergence in green
        plt.plot(np.linspace(1, self.num_epochs, len(self.recon_loss)),
                 self.recon_loss,
                 'r')
        plt.plot(np.linspace(1, self.num_epochs, len(self.kl_loss)),
                 self.kl_loss,
                 'g')

        # Add legend, title
        plt.legend(['Reconstruction', 'Kullback-Leibler'])
        plt.title(self.name)
        plt.show()

    def save_model(self, savepath):
        """ Save save_model state dictionary """
        torch.save(self.model.state_dict(), savepath)

    def load_model(self, loadpath):
        """ Load state dictionary into save_model """
        state = torch.load(loadpath)
        self.model.load_state_dict(state)


if __name__ == "__main__":
    # pic preprocess
    pic_preprocess("raw_pics", "pics", (64, 64))

    mydataset = MyDataSet("pics")
    train_loader = DataLoader(dataset=mydataset,
                              batch_size=1024,
                              shuffle=True,
                              )

    # Init save_model
    model = VAE()

    # Init trainer
    trainer = VAETrainer(model=model,
                         train_iter=train_loader,
                         viz=True)

    # Train
    trainer.train(num_epochs=2500,
                  lr=1e-4,
                  weight_decay=1e-5)
