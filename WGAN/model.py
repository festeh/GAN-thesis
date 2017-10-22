from torch import nn, optim
from torch.autograd import Variable
import torch
from torch.nn import functional as F


class ModelWrapperWGAN:
    def __init__(self, d, g,
                 clamp_lower, clamp_upper,
                 opt_params,
                 input_, noise,
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
        # self.label = label
        self.noise = noise

        self.meters = meters
        self.loggers = loggers
        #
        self.clamp_lower = clamp_lower
        self.clamp_upper = clamp_upper
        # self.REAL_LABEL = 1
        # self.FAKE_LABEL = 0

        self.noise_dim = self.g.noise_dim

    def __call__(self, sample):
        self.d.zero_grad()
        data = sample[0].cuda()
        batch_size = data.size(0)
        self.input.resize_as_(data).copy_(data)
        inputv = Variable(self.input)

        output = self.d(inputv)
        D_x = torch.mean(output)

        # train with fake
        self.noise.resize_(batch_size,
                           self.noise.size(1)).normal_(0, 1)
        fake = self.g(Variable(self.noise))
        output = self.d(fake.detach())
        D_G_z1 = torch.mean(output)

        D_loss = -(D_x - D_G_z1)
        D_loss.backward()

        self.opt_d.step()
        for p in self.d.parameters():
            p.data.clamp_(self.clamp_lower, self.clamp_upper)

        should_train_G = sample[-1]

        D_G_z2 = None
        G_loss = None
        if should_train_G:
            self.g.zero_grad()
            self.noise.resize_(batch_size,
                               self.noise.size(1)).normal_(0, 1)
            fake = self.g(Variable(self.noise))
            output = self.d(fake)
            D_G_z2 = torch.mean(output)
            G_loss = -D_G_z2
            G_loss.backward()
            self.opt_g.step()

        return {"d_loss": D_loss,
                "g_loss": G_loss,
                "real_prob": D_x,
                "fake_prob_before": D_G_z1,
                "fake_prob_after": D_G_z2}, None

    def generate(self, noise):
        return (1.0 + self.g(noise)) / 2.0
