import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision
from model.covmodel import CovGenerator, CovDiscriminator
from model.vitmodel import VITGenerator, VITDiscriminator
from utils import *


class Generator(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.module = ""

        if model_name == "cov":
            self.module = CovGenerator()
        elif model_name == "vit":
            self.module = VITGenerator()

    def forward(self, noise):
        return self.module(noise)


class Discriminator(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.module = ""

        if model_name == "cov":
            self.module = CovDiscriminator()
        elif model_name == "vit":
            self.module = VITDiscriminator()

    def forward(self, img):
        return self.module(img)


class WGPGAN(nn.Module):
    """ Super class to contain both Discriminator (D) and Generator (G)
    """

    def __init__(self, model_name):
        super().__init__()
        self.__dict__.update(locals())
        self.G = Generator(model_name)
        self.D = Discriminator(model_name)
        if model_name == "cov":
            self.z_dim = 100
        elif model_name == "vit":
            self.z_dim = 1024
        else:
            raise "You can choose model_name from cov and vit"


class WGPGANTrainer:
    """ Object to hold data iterators, train a GAN variant
    """

    def __init__(self, model, train_iter, viz=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        self.model = model.to(self.device)
        self.name = model.__class__.__name__

        self.train_iter = train_iter

        self.Glosses = []
        self.Dlosses = []

        self.viz = viz
        self.num_epochs = 0

    def train(self, num_epochs, G_lr=1e-4, D_lr=1e-4, D_steps=5):
        # self.load_model("save_model/save_model.pt")

        # Initialize optimizers, do not use adam
        G_optimizer = optim.RMSprop(params=[p for p in self.model.G.parameters()
                                            if p.requires_grad], lr=G_lr, weight_decay=1e-5)
        D_optimizer = optim.RMSprop(params=[p for p in self.model.D.parameters()
                                            if p.requires_grad], lr=D_lr, weight_decay=1e-5)

        # Approximate steps/epoch given D_steps per epoch
        # --> roughly train in the same way as if D_step (1) == G_step (1)
        epoch_steps = int(np.ceil(len(self.train_iter) / (D_steps)))

        # Begin training
        for epoch in range(1, num_epochs + 1):

            print("***************************************")
            print(epoch)

            self.model.train()
            G_losses, D_losses = [], []

            for _ in tqdm(range(epoch_steps)):

                D_step_loss = []

                for _ in range(D_steps):
                    # Reshape images
                    images = self.process_batch(self.train_iter)

                    # TRAINING D: Zero out gradients for D
                    D_optimizer.zero_grad()

                    # Train the discriminator to approximate the Wasserstein
                    # distance between real, generated distributions
                    D_loss = self.train_D(images)

                    # Update parameters
                    D_loss.backward()
                    D_optimizer.step()

                    # Log results, backpropagate the discriminator network
                    D_step_loss.append(D_loss.item())

                # We report D_loss in this way so that G_loss and D_loss have
                # the same number of entries.
                D_losses.append(np.mean(D_step_loss))

                # TRAINING G: Zero out gradients for G
                G_optimizer.zero_grad()

                # Train the generator to (roughly) minimize the approximated
                # Wasserstein distance
                G_loss = self.train_G(images)

                # Log results, update parameters
                G_losses.append(G_loss.item())
                G_loss.backward()
                G_optimizer.step()

            # Save progress
            # self.Glosses.extend(G_losses)
            # self.Dlosses.extend(D_losses)

            # Progress logging
            print("Epoch[%d/%d], G Loss: %.4f, D Loss: %.4f"
                  % (epoch, num_epochs, np.mean(G_losses), np.mean(D_losses)))
            self.num_epochs += 1

            # Visualize generator progress
            if self.viz:
                self.generate_images(epoch)

            self.save_model(f"save_model/{self.name}_epoch_{epoch}.pt")

    def train_D(self, images, LAMBDA=10):
        """ Run 1 step of training for discriminator

        Input:
            images: batch of images (reshaped to [batch_size, -1])
        Output:
            D_loss: Wasserstein loss for discriminator,
            -E[D(x)] + E[D(G(z))] + λE[(||∇ D(εx + (1 − εG(z)))|| - 1)^2]
        """
        # ORIGINAL CRITIC STEPS:
        # Sample noise, an output from the generator
        noise = self.compute_noise(images.shape[0], self.model.z_dim)
        G_output = self.model.G(noise)

        # Use the discriminator to sample real, generated images
        DX_score = self.model.D(images)  # D(z)
        DG_score = self.model.D(G_output)  # D(G(z))

        # GRADIENT PENALTY:
        # Uniformly sample along one straight line per each batch entry.
        epsilon = torch.rand([images.shape[0], 1, 1, 1]).cuda()

        # Generate images from the noise, ensure unit gradient norm 1
        # See Section 4 and Algorithm 1 of original paper for full explanation.
        G_interpolation = epsilon * images + (1 - epsilon) * G_output
        D_interpolation = self.model.D(G_interpolation)

        # Compute the gradients of D with respect to the noise generated input
        weight = torch.ones(D_interpolation.size()).to(self.device)

        gradients = torch.autograd.grad(outputs=D_interpolation,
                                        inputs=G_interpolation,
                                        grad_outputs=weight,
                                        only_inputs=True,
                                        create_graph=True,
                                        retain_graph=True)[0]

        # Full gradient penalty
        grad_penalty = LAMBDA * torch.mean((gradients.norm(2, dim=1) - 1) ** 2)

        # Compute WGAN-GP loss for D
        D_loss = torch.mean(DG_score) - torch.mean(DX_score) + grad_penalty

        return D_loss

    def train_G(self, images):
        """ Run 1 step of training for generator

        Input:
            images: batch of images reshaped to [batch_size, -1]
        Output:
            G_loss: wasserstein loss for generator,
            -E[D(G(z))]
        """
        # Get noise, classify it using G, then classify the output of G using D.
        noise = self.compute_noise(images.shape[0], self.model.z_dim)  # z
        G_output = self.model.G(noise)  # G(z)
        DG_score = self.model.D(G_output)  # D(G(z))

        # Compute WGAN-GP loss for G (same loss as WGAN)
        G_loss = -1 * (torch.mean(DG_score))

        return G_loss

    def compute_noise(self, batch_size, z_dim):
        """ Compute random noise for input to the Generator G """
        return torch.randn(batch_size, z_dim).to(self.device)

    def process_batch(self, iterator):
        """ Generate a process batch to be input into the discriminator D """
        images = next(iter(iterator))
        images = images.to(self.device)
        return images

    def generate_images(self, epoch, num_outputs=36, save=True):
        """ Visualize progress of generator learning """
        # Turn off any regularization
        self.model.eval()

        # Sample noise vector
        noise = self.compute_noise(num_outputs, self.model.z_dim)

        # Transform noise to image
        images = self.model.G(noise)

        # Save images if desired
        if save:
            outname = 'out/'
            torchvision.utils.save_image(images,
                                         outname + 'recons_%d.png'
                                         % epoch, nrow=int(num_outputs ** 0.5))

    def viz_loss(self):
        """ Visualize loss for the generator, discriminator """
        # Set style, figure size
        plt.style.use('ggplot')
        plt.rcParams["figure.figsize"] = (8, 6)

        # Plot Discriminator loss in red
        plt.plot(np.linspace(1, self.num_epochs, len(self.Dlosses)),
                 self.Dlosses,
                 'r')

        # Plot Generator loss in green
        plt.plot(np.linspace(1, self.num_epochs, len(self.Dlosses)),
                 self.Glosses,
                 'g')

        # Add legend, title
        plt.legend(['Discriminator', 'Generator'])
        plt.title(self.name)
        plt.show()

    def save_model(self, savepath):
        """ Save save_model state dictionary """
        torch.save(self.model.state_dict(), savepath)

    def load_model(self, loadpath):
        """ Load state dictionary into save_model """
        state = torch.load(loadpath)
        self.model.load_state_dict(state)


class WGAN_GP_Test:
    def __init__(self, model, model_path):
        self.model = model
        self.model_path = model_path
        self.name = model.__class__.__name__

    def compute_noise(self, batch_size, z_dim):
        """ Compute random noise for input to the Generator G """
        return torch.randn(batch_size, z_dim)

    def load_model(self, loadpath):
        """ Load state dictionary into save_model """
        state = torch.load(loadpath)
        self.model.load_state_dict(state)

    def test_img(self):
        self.load_model(self.model_path)
        self.model.eval()
        noise = self.compute_noise(36, self.model.z_dim)
        images = self.model.G(noise)
        torchvision.utils.save_image(images, 'test.png', nrow=6)


if __name__ == "__main__":
    # pic preprocess
    pic_preprocess("raw_pics", "pics", (64, 64))

    # Load data
    mydataset = MyDataSet("pics")
    train_loader = DataLoader(dataset=mydataset,
                              batch_size=64,
                              shuffle=True,
                              )

    # Init save_model
    model = WGPGAN(model_name="cov")   # cov or vit

    # Init trainer
    trainer = WGPGANTrainer(model=model,
                            train_iter=train_loader,
                            viz=True)

    # Train
    trainer.train(num_epochs=100,
                  G_lr=8e-2,
                  D_lr=1e-3,
                  D_steps=1)

    # trainer.generate_images(epoch=25, num_outputs=36, save=True)
