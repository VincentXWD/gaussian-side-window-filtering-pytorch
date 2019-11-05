"""
@author: Wendong Xu
@contact: kirai.wendong@gmail.com
@file: gaussian_side_window.py
@time: 2019-11-05 22:36
@desc: 
"""

import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, weight_size, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(weight_size, numbers.Number):
            weight_size = [weight_size] * dim
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        weight_kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in weight_size])
        for size, std, mgrid in zip(weight_size, sigma, meshgrids):
            mean = (size - 1) / 2
            weight_kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        weight_kernel = weight_kernel / torch.sum(weight_kernel)

        kernel = torch.zeros(kernel_size)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
        weight_kernel = weight_kernel.view(1, 1, *weight_kernel.size())
        weight_kernel = weight_kernel.repeat(channels,
                                             *[1] * (weight_kernel.dim() - 1))
        kernel[:, :, 0:weight_size[0], 0:weight_size[1]] = weight_kernel

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.
                format(dim))

    def set_weight(self, weight):
        self.weight = weight

    def forward(self, input):
        return self.conv(input, weight=self.weight, groups=self.groups)


class SideWindowGaussianSmoothing(object):
    """ Side window filtering with gaussian kernel.
    Constructure 8 different directions kernel to do filtering. Use the kernel that with the cloest value to original image.
    It will be iterated several times. More details please refer the paper:
    Yin H, Gong Y, Qiu G. Side Window Filtering[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019: 8758-8766.
    https://arxiv.org/pdf/1905.07177.pdf
    """
    def __init__(self, iteration, radius, sigma=4):
        half_radius = int(math.ceil(radius / 2))
        self.edge = [
            GaussianSmoothing(1,
                              weight_size=[radius, half_radius],
                              kernel_size=[radius, radius],
                              sigma=sigma,
                              dim=2) for _ in range(4)
        ]
        self.corner = [
            GaussianSmoothing(1,
                              weight_size=[half_radius, half_radius],
                              kernel_size=[radius, radius],
                              sigma=sigma,
                              dim=2) for _ in range(4)
        ]
        self.iteration = iteration

        for k in range(1, 4):
            self.edge[k].set_weight(
                torch.rot90(self.edge[0].weight, k=k, dims=(2, 3)))
            self.corner[k].set_weight(
                torch.rot90(self.corner[0].weight, k=k, dims=(2, 3)))

    def __call__(self, x):
        n, c, h, w = x.shape
        diff = torch.zeros(n, 8, h, w, dtype=torch.float)
        y = x.clone()

        for ch in range(c):
            img = x[:, ch, ::].clone().view(n, 1, h, w)
            for _ in range(self.iteration):
                for k in range(4):
                    diff[:, k, ::] = F.interpolate(self.edge[k](img),
                                                   (h, w)) - img
                    diff[:, k + 4, ::] = F.interpolate(self.corner[k](img),
                                                       (h, w)) - img
                abs_diff = torch.abs(diff)
                mask = torch.argmin(abs_diff, dim=1, keepdim=True)
                masked_diff = torch.gather(input=diff, dim=1, index=mask)
                img += masked_diff
            y[:, ch, ::] = img
        return y


if __name__ == '__main__':
    import imageio
    import numpy as np
    img = np.transpose(np.array([imageio.imread('./lena.jpeg')]), [0, 3, 1, 2])
    img = torch.from_numpy(img).float()

    gaussian = SideWindowGaussianSmoothing(iteration=2, radius=5, sigma=3)

    img = gaussian(img).numpy()[0]
    img = np.transpose(img, [1, 2, 0]).astype('uint8')
    imageio.imwrite('./lena_out.jpeg', img)

