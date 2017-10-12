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
    model_wrapper = state['network']
    for meter_name, meter in model_wrapper.meters.items():
        meter_loss = state['network'].meters[meter_name]
        meter_loss.add(state['loss'][meter_name].data[0])

def on_start_epoch(state):
    reset_meters(state['network'].meters)
    state['iterator'] = tqdm(state['iterator'])

def on_end_epoch(state):
    model_wrapper = state['network']
    for meter_name, meter in model_wrapper.meters.items():
        train_loss_logger = model_wrapper.loggers[meter_name]
        train_loss_logger.log(state['epoch'], meter.value()[0])
    noise_sample = Variable(torch.randn(64, model_wrapper.noise_dim)).cuda()
    img = get_image(model_wrapper, noise_sample)
    model_wrapper.loggers["generated_image"].log(img)


class GanEngine(Engine):
    def __init__(self):
        super().__init__()
        self.hooks['on_sample'] = on_sample
        self.hooks['on_forward'] = on_forward
        self.hooks['on_start_epoch'] = on_start_epoch
        self.hooks['on_end_epoch'] = on_end_epoch

    def train(self, network, iterator, maxepoch, optimizer=None):
        state = {
                'network': network,
                'iterator': iterator,
                'maxepoch': maxepoch,
                'epoch': 0,
                't': 0,
                'train': True,
                }

        self.hook('on_start', state)
        while state['epoch'] < state['maxepoch']:
            self.hook('on_start_epoch', state)
            for sample in state['iterator']:
                state['sample'] = sample
                self.hook('on_sample', state)

                def closure():
                    loss, output = state['network'](state['sample'])
                    state['output'] = output
                    state['loss'] = loss
                    self.hook('on_forward', state)
                    # to free memory in save_for_backward
                    state['output'] = None
                    state['loss'] = None
                    return loss

                closure()
                self.hook('on_update', state)
                state['t'] += 1
            state['epoch'] += 1
            self.hook('on_end_epoch', state)
        self.hook('on_end', state)
        return state