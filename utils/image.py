from torchvision.utils import make_grid
import numpy as np
from PIL.Image import fromarray


def get_image(model_wrapper, noise):
    gen_images = model_wrapper.generate(noise).cpu()
    visdom_img = make_grid(gen_images.data.view(-1, 1, 28, 28)).numpy()
    visdom_img = np.transpose(visdom_img, (1, 2, 0))[:, :, 0]
    return visdom_img
