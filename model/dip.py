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
from torchvision import transforms
from PIL import Image
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from kornia import feature
import numpy as np
import matplotlib
import cv2 as cv
import torch

matplotlib.use('Agg')


def load_image(full_path, input_size=(240., 320.)):
    # Read image
    img_np = cv.cvtColor(cv.imread(full_path, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)

    # Scaling that keeps the aspect ratio
    h, w, c = img_np.shape
    h_s = input_size[0] / float(h)
    w_s = input_size[1] / float(w)
    s = max(h_s, w_s)

    # Center crop
    center_h = int(h / 2)
    center_new_h = int(int(input_size[0] / s) / 2)
    center_w = int(w / 2)
    center_new_w = int(int(input_size[1] / s) / 2)
    img_np = img_np[center_h - center_new_h:center_h + center_new_h, center_w - center_new_w:center_w + center_new_w]

    return img_np


def tensor2numpy_image(x):
    return np.asarray(np.clip(x.permute(1, 2, 0).numpy() * 255, 0, 255), dtype=np.uint8)


def plot_images(fig_x, fig_y, images):
    fig = plt.figure(figsize=(fig_y * 2, fig_x * 3))

    for i, im in enumerate(images):
        plt.subplot(fig_x, fig_y, i + 1)
        plt.imshow(im)

    return fig


def perspective_transform(keypoints, homography):
    """
        keypoints: [[y_0, x_0], ..., [y_n, x_n]]
    """

    homogeneous_keypoints = np.concatenate([keypoints, np.ones((keypoints.shape[0], 1))], axis=1)
    warped_keypoints = np.dot(homogeneous_keypoints, np.transpose(homography))

    return warped_keypoints[:, :2] / warped_keypoints[:, 2:]


def draw_keypoints(image, keypoints, color=(0, 255, 0)):
    for keypoint in keypoints:
        x, y = int(keypoint[0]), int(keypoint[1])
        cv.circle(image, (y, x), 3, color, 1)


def sample_pairs(x, min_size, padding_color='black', padding=False, negative_sampling=True):
    # Original image size
    x_w, x_h = x.size
    aspect_ratio = float(x_h) / float(x_w)

    # Minimum crop size
    x_w_min = int(x_w * min_size)
    x_h_min = int(x_h * min_size)

    # Computing padding
    if padding:
        x_w_padding = x_w_min // 2
        x_h_padding = x_h_min // 2

        # Image with padding
        x_padding = Image.new(x.mode, (x_w + x_w_padding, x_h + x_h_padding), padding_color)
        x_padding.paste(x, (x_w_padding, x_h_padding))
        x_padding_w, x_padding_h = x_padding.size
    else:
        x_padding = Image.new(x.mode, (x_w, x_h), padding_color)
        x_padding.paste(x, (0, 0))
        x_padding_w, x_padding_h = x_padding.size

    # Reference crop W
    ref_w_min = np.random.randint(0, x_padding_w - x_w_min)
    ref_w_max = np.random.randint(ref_w_min + x_w_min, x_padding_w)
    ref_w = ref_w_max - ref_w_min

    # Reference crop H. Keep aspect ratio
    ref_h = int(ref_w * aspect_ratio)
    ref_h_min = np.random.randint(0, x_padding_h - ref_h)
    ref_h_max = ref_h_min + ref_h

    try:
        # Positive target crop W
        pos_w_min = np.random.randint(np.maximum(ref_w_min - x_w_min, 0), np.minimum(ref_w_max + x_w_min, x_padding_w))

        if pos_w_min >= ref_w_max:
            pos_w_max = pos_w_min
            pos_w_min = np.random.randint(np.maximum(ref_w_min - x_w_min, 0), pos_w_max - x_w_min)
        else:
            pos_w_max = np.random.randint(pos_w_min + x_w_min, x_padding_w)
        pos_w = pos_w_max - pos_w_min

        # Positive target crop H. Keep aspect ratio
        pos_h = int(pos_w * aspect_ratio)
        pos_h_min = np.random.randint(np.maximum(ref_h_min - pos_h, 0), ref_h_max - pos_h)
        pos_h_max = pos_h_min + pos_h
    except ValueError:
        pos_w_min = ref_w_min
        pos_w_max = ref_w_max
        pos_h_min = ref_h_min
        pos_h_max = ref_h_max

    if negative_sampling:
        try:
            # Negative target crop W
            if ref_w_min > (x_padding_w - ref_w_max):
                neg_w_min = np.random.randint(0, ref_w_min - x_w_min)
                neg_w_max = np.random.randint(neg_w_min + x_w_min, ref_w_min)
            else:
                neg_w_min = np.random.randint(ref_w_max, x_padding_w - ref_w_min)
                neg_w_max = np.random.randint(neg_w_min + ref_w_min, x_padding_w)
            neg_w = neg_w_max - neg_w_min

            # Positive target crop H. Keep aspect ratio
            neg_h = int(neg_w * aspect_ratio)

            if ref_h_min > (x_padding_h - ref_h_max):
                neg_h_min = np.random.randint(0, ref_h_min - neg_h)
                assert neg_h_min + neg_h <= ref_h_min
            else:
                neg_h_min = np.random.randint(ref_h_max, x_padding_h - neg_h)
                assert neg_h_min + neg_h <= x_padding_h
            neg_h_max = neg_h_min + neg_h
        except (ValueError, AssertionError):
            neg_w_min = None
            neg_w_max = None
            neg_h_min = None
            neg_h_max = None
    else:
        neg_w_min = None
        neg_w_max = None
        neg_h_min = None
        neg_h_max = None

    # Cropping
    x_ref = x_padding.crop((ref_w_min, ref_h_min, ref_w_max, ref_h_max))
    x_pos = x_padding.crop((pos_w_min, pos_h_min, pos_w_max, pos_h_max))
    x_neg = None if (neg_w_min is None) else x_padding.crop((neg_w_min, neg_h_min, neg_w_max, neg_h_max))

    # Size validation
    x_pos_w, x_pos_h = x_pos.size
    if (x_pos_w < x_w_min) or (x_pos_h < x_h_min):
        x_pos = x_ref
    if not (x_neg is None):
        x_neg_w, x_neg_h = x_neg.size
        if (x_neg_w < x_w_min) or (x_neg_h < x_h_min):
            x_neg = None

    return x_ref, x_pos, x_neg


def normalize_gradients(x):
    x_norm = (x - np.mean(x)) / (np.std(x) + 1e-5)
    x_norm = np.swapaxes(np.swapaxes(x_norm, 0, 1), 1, 2)
    x_norm = np.max(np.abs(x_norm), axis=2)
    x_norm /= np.max(x_norm)
    return x_norm


def detect_classical(grads, hysteresis_threshold=(0.0, 0.0), nms=(3, 3), top_k=1000):
    # NMS from Kornia
    grads_pth = torch.tensor(grads)
    grads_pth = grads_pth.unsqueeze(0).unsqueeze(0)
    grads_pth = feature.non_maxima_suppression2d(grads_pth, (nms[0], nms[1]))

    # Convert to numpy
    grads = grads_pth[0][0].cpu().data.numpy()

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

    return np.hstack((keypoints, keypoints_likelihood.reshape(-1, 1)))


def detect_keypoints(input_image, guided_backprop, output_neuron_idx, hysteresis_threshold=(0.0, 0.0), nms=(3, 3), top_k=500):
    # Guided back-propagated activations
    grads = guided_backprop(input_image, output_neuron_idx)

    # Normalize the raw gradients
    grads_np = normalize_gradients(grads.cpu().data.numpy())

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

    return np.hstack((keypoints, keypoints_likelihood.reshape(-1, 1)))


def crop_rescale_local_patches(input_image, keypoints, patch_size, rescaled_size, gray_scale):
    to_gray = None
    if gray_scale:
        to_gray = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
    local_patches = []
    for lc_x, lc_y, _ in keypoints:
        lc_x = int(np.maximum(0, lc_x - (patch_size[0] // 2)))
        lc_y = int(np.maximum(0, lc_y - (patch_size[1] // 2)))
        crop_tensor = torch.nn.functional.interpolate(input_image[:, :, lc_x:lc_x + patch_size[0], lc_y:lc_y + patch_size[1]], rescaled_size, mode='bicubic', align_corners=False)[0]
        if gray_scale:
            crop_tensor = to_gray(crop_tensor.cpu())
        local_patches.append(crop_tensor)
    return torch.stack(local_patches)


def extract_descriptions(input_image, keypoints, network, device, batch_size=1, patch_size=(18, 24), input_size=(24, 32), runs_hard_net=False):
    # To return
    descriptions = []

    # Crop local patches
    local_patches = crop_rescale_local_patches(input_image, keypoints, patch_size, input_size, runs_hard_net)

    # Compute descriptions
    batch_idx = 0
    local_patches_size = local_patches.shape[0]
    while batch_idx < local_patches_size:
        input_batch = local_patches[batch_idx:batch_idx + batch_size, :, :, :]
        batch_idx += batch_size
        input_batch = input_batch.to(device)

        if runs_hard_net:
            z = network(input_batch)
        else:
            _, z = network(input_batch)

        descriptions.append(z.detach().cpu().numpy())

    return np.vstack(descriptions).astype(np.float32)  #, local_patches


def apply_hysteresis_threshold(image, low, high):
    # Ensure low always below high
    low = np.clip(low, a_min=None, a_max=high)
    mask_low = image > low
    mask_high = image > high

    # Connected components of mask_low
    labels_low, num_labels = ndi.label(mask_low)

    # Check which connected components contain pixels from mask_high
    sums = ndi.sum(mask_high, labels_low, np.arange(num_labels + 1))
    connected_to_high = sums > 0
    thresholded = connected_to_high[labels_low]

    return thresholded


def find_correspondences(descriptions, cross_check=True):
    # BF matches
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=cross_check)
    correspondences = bf.match(descriptions[0], descriptions[1])

    return correspondences


def compute_homography(keypoints, matches, min_match_count=5, ransac_threshold=3):
    # To return
    homography, mask = None, None

    # Compute homography
    if len(matches) > min_match_count:
        src_pts = np.float32([keypoints[0][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[1][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        homography, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, ransac_threshold)
    else:
        print('Not enough matches.')

    return homography, mask
