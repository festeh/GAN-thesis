import argparse
import torch
import torch.utils.data
from torchvision import datasets, transforms
import torchnet as tnt
from torchnet.logger import VisdomPlotLogger, VisdomLogger
from DCGAN.model import ModelWrapperDCGAN, DiscriminatorDCGAN, GeneratorDCGAN, weights_init
from utils.gan_engine import GanEngine
from utils.image import get_image
from torch.autograd import Variable


parser = argparse.ArgumentParser(description='VAE')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--noise-dim', type=int, default=30, metavar='N',
                    help='input noise dimension (default: 20)')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate for Generator, default=0.00005')
parser.add_argument('--beta', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--env', type=str, default="main", metavar='E',
                    help='which env is used in Visdom')

args = parser.parse_args()
torch.cuda.manual_seed(args.seed)


def get_iterator(mode):
    dataset = datasets.MNIST('../data',
                             train=mode,
                             download=True,
                             transform=transforms.Compose(
                                 [transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=1, pin_memory=True)
    return loader


meter_d_loss = tnt.meter.AverageValueMeter()
meter_g_loss = tnt.meter.AverageValueMeter()
d_loss_logger = VisdomPlotLogger('line',
                                     opts={'title': 'Discriminator Loss'},
                                     env=args.env)
g_loss_logger = VisdomPlotLogger('line',
                                     opts={'title': 'Generator Loss'},
                                     env=args.env)
image_logger = VisdomLogger('image', env=args.env)

discr = DiscriminatorDCGAN(1, 16, 28 * 28)
discr.apply(weights_init)
discr.cuda()

gen = GeneratorDCGAN(args.noise_dim , 16, 28 * 28).cuda()
gen.apply(weights_init)
gen.cuda()

input_ = torch.FloatTensor(args.batch_size, 1, 28, 28).cuda()
noise = torch.FloatTensor(args.batch_size, args.noise_dim).cuda()
label = torch.FloatTensor(args.batch_size).cuda()

model_wrapper = ModelWrapperDCGAN(g=gen, d=discr,
                                  opt_params={"lrG": args.lrG,
                                              "lrD":args.lrD,
                                              "beta":args.beta},
                                  input_=input_, noise=noise, label=label,
                                  meters={"d_loss": meter_d_loss,
                                          "g_loss": meter_g_loss},
                                  loggers={"d_loss": d_loss_logger,
                                           "g_loss": g_loss_logger,
                                           "generated_image": image_logger})

engine = GanEngine()

engine.train(network=model_wrapper,
             iterator=get_iterator(True),
             maxepoch=args.epochs, optimizer=None)
with open(f"models/{args.env}_g", "wb") as f:
    torch.save(model_wrapper.g , f)
with open(f"models/{args.env}_d", "wb") as f:
    torch.save(model_wrapper.d , f)
# VisdomSaver(["main", ]).save()
