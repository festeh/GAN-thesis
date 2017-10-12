from torch import nn, optim
from torch.autograd import Variable
import torch
from torch.nn import functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class GeneratorDCGAN(nn.Module):
    def __init__(self, noise_dim, n_filters, img_dim):
        super(GeneratorDCGAN, self).__init__()
        self.noise_dim = noise_dim
        self.n_filters = n_filters
        self.img_dim = img_dim

        self.linear = torch.nn.Linear(self.noise_dim,
                                      self.noise_dim * 7 * 7)
        self.relu = torch.nn.ReLU(True)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.noise_dim, n_filters, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters),
            nn.ReLU(True),
            # state size. (n_filters*2) x 16 x 16
            # nn.ConvTranspose2d(n_filters * 2, n_filters, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(n_filters),
            # nn.ReLU(True),
            # state size. (n_filters) x 32 x 32
            nn.ConvTranspose2d(n_filters, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        reshaped_input = self.relu(self.linear(input)).view(
            -1, self.noise_dim, 7, 7)
        output = self.main(reshaped_input)
        return output


class DiscriminatorDCGAN(nn.Module):
    def __init__(self, n_channels, n_filters, img_dim):
        super(DiscriminatorDCGAN, self).__init__()
        self.n_channels = n_channels
        self.n_filters = n_filters
        self.img_dim = img_dim
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.n_channels, self.n_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (n_filters) x 32 x 32
            nn.Conv2d(self.n_filters, self.n_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (n_filters*2) x 16 x 16
            nn.Conv2d(n_filters * 2, n_filters * 4, 4, 2, 2, bias=False),
            nn.BatchNorm2d(n_filters * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # # state size. (n_filters*4) x 8 x 8
            # nn.Conv2d(n_filters * 4, n_filters * 8, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(n_filters * 8),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. (n_filters*8) x 4 x 4
            nn.Conv2d(n_filters * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)



class ModelWrapperDCGAN:
    def __init__(self, d, g,
                 opt_params,
                 input_, label, noise,
                 meters, loggers):
        self.d = d
        self.g = g
        lrD = opt_params["lrD"]
        lrG = opt_params["lrG"]
        beta = opt_params["beta"]
        self.opt_d = optim.Adam(d.parameters(),
                                lr=lrD, betas=(beta, 0.999))
        self.opt_g = optim.Adam(g.parameters(),
                                lr=lrG, betas=(beta, 0.999))

        self.input = input_
        self.label = label
        self.noise = noise

        self.meters = meters
        self.loggers = loggers

        self.REAL_LABEL = 1
        self.FAKE_LABEL = 0

        self.noise_dim = self.g.noise_dim

    def __call__(self, sample):
        self.d.zero_grad()
        data = sample[0]
        batch_size = data.size(0)
        data = data.cuda()
        self.input.resize_as_(data).copy_(data)
        self.label.resize_(batch_size).fill_(self.REAL_LABEL)
        inputv = Variable(self.input)
        labelv = Variable(self.label)

        output = self.d(inputv)
        errD_real = F.binary_cross_entropy(output, labelv)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        self.noise.resize_(batch_size,
                           self.noise.size(1)).normal_(0, 1)
        noisev = Variable(self.noise)
        fake = self.g(noisev)
        labelv = Variable(self.label.fill_(self.FAKE_LABEL))
        output = self.d(fake.detach())
        errD_fake = F.binary_cross_entropy(output, labelv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        self.opt_d.step()

        self.g.zero_grad()
        labelv = Variable(self.label.fill_(self.REAL_LABEL))  # fake labels are real for generator cost
        output = self.d(fake)
        errG = F.binary_cross_entropy(output, labelv)
        errG.backward()
        D_G_z2 = output.data.mean()
        self.opt_g.step()

        return {"d_loss": errD,
                "g_loss": errG,
                "real_prob": D_x,
                "fake_prob_before": D_G_z1,
                "fake_prob_after": D_G_z2}, None

    def generate(self, noise):
        return (1.0 + self.g(noise)) / 2.0

