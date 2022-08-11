# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
...
"""

__author__ = "..."
__email__ = "..."
__license__ = "..."
__version__ = "1.0"

# External modules
import torch
from torch.autograd import Function


class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(ctx, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        ctx.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_saved_tensors, output = ctx.saved_tensors

        positive_mask_1 = (input_saved_tensors > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)

        grad_input = torch.addcmul(torch.zeros(input_saved_tensors.size()).type_as(input_saved_tensors),
                                   torch.addcmul(torch.zeros(input_saved_tensors.size()).type_as(input_saved_tensors), grad_output, positive_mask_1),
                                   positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, network, device):
        self.network = network
        self.network = network.to(device)
        self.network.eval()
        self.device = device

        def recursive_relu_apply(modules):
            for idx, module in modules._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    modules._modules[idx] = GuidedBackpropReLU.apply

        recursive_relu_apply(self.network)

    def forward(self, input_image):
        h, z = self.network(input_image)
        return h

    def __call__(self, input_image, idx=None):
        # Predict input image
        output = self.forward(input_image)

        # Initialize one hot output mask
        if idx is None:
            idx = torch.argmax(output)
        one_hot = torch.zeros(output.size())
        one_hot[0][idx] = 1
        one_hot = one_hot.to(self.device)
        one_hot.requires_grad = True

        # Apply mask and back-propagate the activation to the input space
        one_hot = torch.neg(torch.sum(one_hot * output))
        one_hot.backward()

        # Return the input gradients w.r.t. the output neuron
        return input_image.grad[0, :, :, :]
