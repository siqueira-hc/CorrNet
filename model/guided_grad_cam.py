# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    This file contains the functions for detecting keypoints with the proposed alternative to grad-CAM (Class Activation Map) [1].

    Reference:
        [1] Selvaraju, R.R., Cogswell, M., Das, A., Vedantam, R., Parikh, D. and Batra, D., 2017. Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE international conference on computer vision (pp. 618-626).
"""

__author__ = "..."
__email__ = "..."
__license__ = "..."
__version__ = "1.0"

# External modules
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
import torch.nn.functional as F
from kornia import feature
import numpy as np
import torch

# Internal modules
from model.dip import normalize_gradients, apply_hysteresis_threshold


def get_activation_maps(input_image1, input_image2, corrnet, resize_to_input=True):
    """

    This function returns activation maps for each of the 2 images. Each activation map highlights regions which
    seem similar to the other images content.

    It is using the feature tensor f before the average pooling and the image descriptor h directly after the
    average pooling. Given f1, h1 and f2,

    Arguments:
        input_image1 (torch.Tensor): Reference image in the shape (1, 3, height, width)
        input_image2 (torch.Tensor): Target image in the shape (1, 3, height, width). Shape can be
            different from input_image1
        corrnet (CorrNet): Instance of CorrNet.
        resize_to_input (bool): If true, the returned activation map will have the same width and height as the input
            images. If False, the activation maps will have the width and height of each image's feature map before
            the average pooling is applied

    Returns:
        Tuple of two activation maps (reference_map, target_map). Each map is a numpy array in the shape of (height, width)

    """
    with torch.no_grad():

        # get features
        f1 = corrnet.f(input_image1)
        f2 = corrnet.f(input_image2)
        _, channels, height1, width1 = f1.shape
        _, channels, height2, width2 = f2.shape

        # flatten f for multiplication later
        f1_flat = f1.reshape(channels, height1 * width1)
        f2_flat = f2.reshape(channels, height2 * width2)

        # get h
        h1 = F.normalize(torch.mean(f1_flat, dim=1), dim=0)
        h2 = F.normalize(torch.mean(f2_flat, dim=1), dim=0)

        # multiply each cell in f from one image with h of the other image
        f1_activation = torch.mean((f1_flat * h2.reshape(channels, 1)).reshape(channels, height1, width1), dim=0)
        f2_activation = torch.mean((f2_flat * h1.reshape(channels, 1)).reshape(channels, height2, width2), dim=0)

        # normalization the activations to a range of [0, 1]
        f1_activation_normalized = f1_activation - torch.min(f1_activation)
        f2_activation_normalized = f2_activation - torch.min(f2_activation)
        f1_activation_normalized /= torch.max(f1_activation_normalized)
        f2_activation_normalized /= torch.max(f2_activation_normalized)

        # get numpy
        activation_map1 = f1_activation.cpu().data.numpy()
        activation_map2 = f2_activation.cpu().data.numpy()

        if resize_to_input:
            # resize the activation map to the image size and apply a gaussian kernel
            activation_map1 = resize(activation_map1, (input_image1.shape[2], input_image2.shape[3]))
            activation_map2 = resize(activation_map2, (input_image2.shape[2], input_image2.shape[3]))
            activation_map1 = gaussian_filter(activation_map1, sigma=min(input_image1.shape[2] // height1, input_image1.shape[3] // width1))
            activation_map2 = gaussian_filter(activation_map2, sigma=min(input_image2.shape[2] // height2, input_image2.shape[3] // width2))

    return activation_map1, activation_map2


def detect_keypoints(input_image1, input_image2, corrnet, guided_backprop, output_neuron_idx, hysteresis_threshold=(0.0, 0.0), nms=(3, 3), top_k=1000):
    """
    Arguments:
        input_image1 (torch.Tensor): Reference image in the shape (1, 3, height, width)
        input_image2 (torch.Tensor): Target image in the shape (1, 3, height, width)
        corrnet (CorrNet): CorrNet instance
        guided_backprop: Guided backpropagation instance of cornet
        output_neuron_idx: index of the highest activated neuron of h
        hysteresis_threshold: ...
        nms: ...
        top_k: Take k best keypoints of each image

    Returns:
        The keypoints and gradient maps in the following format:
            [(keypoints1, grads1), (keypoints2, grads2)]

        Keypoints are in the shape (number_of_keypoints, 3) with the values being [y, x, likelihood]
        Gradients are in the shape (height, width) and have the same resolution as their corresponding input images

    """

    # Get gradients
    grads1 = guided_backprop(input_image1, output_neuron_idx)
    grads2 = guided_backprop(input_image2, output_neuron_idx)
    grads_np1 = grads1.cpu().data.numpy()
    grads_np2 = grads2.cpu().data.numpy()

    # Multiply the gradients with the activation map if wanted
    activation1, activation2 = get_activation_maps(input_image1, input_image2, corrnet)
    grads_np1 *= np.expand_dims(activation1, 0)
    grads_np2 *= np.expand_dims(activation2, 0)

    # Normalize the gradients
    grads_np1 = normalize_gradients(grads_np1)
    grads_np2 = normalize_gradients(grads_np2)

    # Apply post processing and extract keypoints from each gradient map
    out = []
    for grads_np, activations_np in [(grads_np1, activation1), (grads_np2, activation2)]:
        # NMS from Kornia
        if nms[0] > 0:
            grads_pth = torch.tensor(grads_np)
            grads_pth = grads_pth.unsqueeze(0).unsqueeze(0)
            grads_pth = feature.non_maxima_suppression2d(grads_pth, (nms[0], nms[1]))

            # Convert to numpy
            grads = grads_pth[0][0].cpu().data.numpy()
        else:
            grads = grads_np

        # Hysteresis threshold
        hysteresis_threshold_min, hysteresis_threshold_max = hysteresis_threshold
        grads_binary = apply_hysteresis_threshold(grads, hysteresis_threshold_min, hysteresis_threshold_max)

        # Keypoints
        keypoints = np.swapaxes(np.vstack(np.nonzero(grads_binary)), 0, 1)
        # Get likelihood based on gradients
        keypoints_likelihood = np.array([grads[k[0]][k[1]] for k in keypoints])

        # Get top keypoints based on the likelihoods
        if top_k > 0.0:
            idx_sort = np.argsort(keypoints_likelihood)[-top_k:]
            keypoints = keypoints[idx_sort]
            keypoints_likelihood = keypoints_likelihood[idx_sort]

        out.append(np.hstack((keypoints, keypoints_likelihood.reshape(-1, 1))))

    return out
