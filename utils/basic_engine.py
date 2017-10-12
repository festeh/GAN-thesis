import torch
from torchnet.engine import Engine
from tqdm import tqdm
from torch.autograd import Variable
from utils.image import get_image

def reset_meters(meters):
    for meter in meters.values():
        meter.reset()

def on_sample(state):
    state['sample'].append(state['train'])

def on_forward(state):
    meter_loss = state['network'].meters["loss"]
    meter_loss.add(state['loss'].data[0])

def on_start_epoch(state):
    reset_meters(state['network'].meters)
    state['iterator'] = tqdm(state['iterator'])

def on_end_epoch(state):
    model_wrapper = state['network']
    meter_loss = model_wrapper.meters["loss"]
    print('Training loss: %.4f' % (meter_loss.value()[0]))
    train_loss_logger = model_wrapper.loggers["train_loss"]
    train_loss_logger.log(state['epoch'], meter_loss.value()[0])

    reset_meters(model_wrapper.meters)

    model_wrapper.eval()

    noise_sample = Variable(torch.randn(64, model_wrapper.noise_dim)).cuda()
    img = get_image(model_wrapper, noise_sample)
    model_wrapper.loggers["generated_image"].log(img)

    BasicEngine().test(model_wrapper, model_wrapper.dataset_iter(False))
    test_loss_logger = model_wrapper.loggers["test_loss"]
    test_loss_logger.log(state['epoch'], meter_loss.value()[0])
    print('Testing loss: %.4f' % (meter_loss.value()[0]))
    model_wrapper.train()


class BasicEngine(Engine):
    def __init__(self):
        super().__init__()
        self.hooks['on_sample'] = on_sample
        self.hooks['on_forward'] = on_forward
        self.hooks['on_start_epoch'] = on_start_epoch
        self.hooks['on_end_epoch'] = on_end_epoch