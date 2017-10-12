from torch import nn
from torch.autograd import Variable
import torch
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, input_size, H_1, H_2, batch_size):
        super(VAE, self).__init__()

        self.input_size = input_size
        self.batch_size = batch_size
        self.noise_dim = H_2

        self.fc1 = nn.Linear(input_size, H_1)
        self.fc21 = nn.Linear(H_1, H_2)
        self.fc22 = nn.Linear(H_1, H_2)
        self.fc3 = nn.Linear(H_2, H_1)
        self.fc4 = nn.Linear(H_1, input_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

    def generate(self, z):
        return self.decode(z)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.input_size))
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= self.batch_size * self.input_size
        return BCE + KLD


class ModelWrapperVAE:

    def __init__(self, model, dataset_iter, meters, loggers):
        self.model = model
        self.dataset_iter = dataset_iter
        self.meters = meters
        self.loggers = loggers
        self.noise_dim = self.model.noise_dim

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def generate(self, noise):
        return self.model.generate(noise)

    def __call__(self, sample):
        x = Variable(sample[0]).cuda()
        recon_batch, mu, logvar = self.model(x)
        loss = self.model.loss(recon_batch, x, mu, logvar)
        return loss, (recon_batch, mu, logvar)
