import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torchvision import datasets, transforms
import torchnet as tnt
from torchnet.logger import VisdomPlotLogger, VisdomLogger, VisdomSaver
from VAE.model import VAE, ModelWrapperVAE
from utils.basic_engine import BasicEngine


parser = argparse.ArgumentParser(description='VAE')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--noise-dim', type=int, default=30, metavar='N',
                    help='input noise dimension (default: 20)')
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
                                 [transforms.ToTensor(), ]))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=1, pin_memory=True)
    return loader


engine = BasicEngine()
meter_loss = tnt.meter.AverageValueMeter()
train_loss_logger = VisdomPlotLogger('line',
                                     opts={'title': 'Train Loss'},
                                     env=args.env)
test_loss_logger = VisdomPlotLogger('line', opts={'title': 'Test Loss'},
                                    env=args.env)
image_logger = VisdomLogger('image', env=args.env)

model = VAE(784, 400, args.noise_dim, batch_size=args.batch_size)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
model.train()
model_wrapper = ModelWrapperVAE(model,
                                dataset_iter=get_iterator,
                                meters={"loss": meter_loss},
                                loggers={"train_loss":train_loss_logger,
                                         "test_loss": test_loss_logger,
                                         "generated_image": image_logger})
engine.train(model_wrapper,
             get_iterator(True), maxepoch=args.epochs, optimizer=optimizer)
with open(f"models/{args.env}", "wb") as f:
    torch.save(model, f)
# VisdomSaver(["main", ]).save()
