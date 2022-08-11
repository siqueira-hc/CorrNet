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
from torch.utils.data import Dataset
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2 as cv
import torch
import os

# Internal modules
from model import dip


class HPatches(Dataset):
    """
        This class loads the HPatches dataset [1].

        Reference:
            [1] Balntas, V., Lenc, K., Vedaldi, A. and Mikolajczyk, K., 2017. HPatches: A benchmark and evaluation of handcrafted and learned local descriptors. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 5173-5182).
    """

    _IMAGE_EXT = '.ppm'
    __NUM_SEQUENCES = 6

    def __init__(self, dataset_folder, transform, input_size=(240., 320.)):
        # Variables
        self.data = [[] for _ in range(HPatches.__NUM_SEQUENCES)]
        self.homography = [[] for _ in range(HPatches.__NUM_SEQUENCES - 1)]
        self.dataset_folder = dataset_folder
        self.transform = transform
        self._sequence_idx = None
        self.h, self.w = input_size

        # Load dataset
        self._load()

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, idx):
        # Get image
        img = self.data[self._sequence_idx][idx]

        # Apply transforms
        to_return_img = self.transform(img)

        # Select homography matrix
        to_return_homography_matrix = self.homography[self._sequence_idx - 1][idx] if self._sequence_idx > 0 else np.eye(3)

        return to_return_img, to_return_homography_matrix

    def set_sequence_idx(self, idx):
        """
            Set the sequence index for sampling.
        """
        self._sequence_idx = idx

    # TODO: Convert to torch
    @staticmethod
    def _compute_homography(H, ref_size, tar_size, target_size):
        """
            Code extracted from: https://github.com/rpautrat/SuperPoint.
        """

        source_size = tf.cast(ref_size, tf.float32)
        source_warped_size = tf.cast(tar_size, tf.float32)

        # target_size = tf.cast(tf.convert_to_tensor(config['preprocessing']['resize']), tf.float32)
        target_size = tf.cast(tf.convert_to_tensor(target_size), tf.float32)

        # Compute the scaling ratio due to the resizing for both images
        s = tf.reduce_max(tf.divide(target_size, source_size))

        # up_scale = tf.diag(tf.stack([1. / s, 1. / s, tf.constant(1.)]))
        up_scale = tf.compat.v1.diag(tf.stack([1. / s, 1. / s, tf.constant(1.)]))

        warped_s = tf.reduce_max(tf.divide(target_size, source_warped_size))

        # down_scale = tf.diag(tf.stack([warped_s, warped_s, tf.constant(1.)]))
        down_scale = tf.compat.v1.diag(tf.stack([warped_s, warped_s, tf.constant(1.)]))

        # Compute the translation due to the crop for both images
        pad_y = tf.cast(((source_size[0] * s - target_size[0]) / tf.constant(2.0)), tf.int32)
        pad_x = tf.cast(((source_size[1] * s - target_size[1]) / tf.constant(2.0)), tf.int32)
        translation = tf.stack([tf.constant(1), tf.constant(0), pad_x,
                                tf.constant(0), tf.constant(1), pad_y,
                                tf.constant(0), tf.constant(0), tf.constant(1)])
        translation = tf.cast(tf.reshape(translation, [3, 3]), tf.float32)
        pad_y = tf.cast(((source_warped_size[0] * warped_s - target_size[0])
                             / tf.constant(2.0)), tf.int32)
        pad_x = tf.cast(((source_warped_size[1] * warped_s - target_size[1])
                             / tf.constant(2.0)), tf.int32)
        warped_translation = tf.stack([tf.constant(1), tf.constant(0), -pad_x,
                                       tf.constant(0), tf.constant(1), -pad_y,
                                       tf.constant(0), tf.constant(0), tf.constant(1)])
        warped_translation = tf.cast(tf.reshape(warped_translation, [3, 3]), tf.float32)

        H = warped_translation @ down_scale @ H @ up_scale @ translation

        return H

    def _load(self):
        sequences = np.sort(np.array(os.listdir(self.dataset_folder)))

        # Iterate over sequences
        for seq in sequences:
            path_to_images = os.path.join(self.dataset_folder, seq)
            files = np.sort(np.array((os.listdir(path_to_images))))

            # Get image file names for each sequence
            images = []
            for file in files:
                if HPatches._IMAGE_EXT in file:
                    images.append(file)

            # Iterate over images
            ref_size = None
            for image_idx in range(HPatches.__NUM_SEQUENCES):
                # Read image
                full_path_to_image = os.path.join(path_to_images, images[image_idx])
                img_np = cv.cvtColor(cv.imread(full_path_to_image, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)

                # Compute scaling factor that keeps the aspect ratio
                h, w, c = img_np.shape
                h_scale = self.h / float(h)
                w_scale = self.w / float(w)
                scale = max(h_scale, w_scale)

                # Crop image to the target input size
                img_np = img_np[:int(self.h / scale), :int(self.w / scale)]
                self.data[image_idx].append(Image.fromarray(np.uint8(img_np)))

                # Compute new homography matrix
                if image_idx == 0:
                    ref_size = img_np.shape[:2]
                else:
                    tar_size = img_np.shape[:2]
                    H = np.loadtxt(os.path.join(path_to_images, 'H_{}_{}'.format(1, image_idx + 1))).astype(np.float)

                    # TODO: Validate torch homo and tf homo
                    # new_H_tf = HPatches._compute_homography(H, ref_size, tar_size, [self.h, self.w])
                    scaled_homo_matrix = HPatches._compute_homography(H, ref_size, tar_size, [self.h, self.w])
                    scaled_homo_matrix = scaled_homo_matrix.numpy()

                    self.homography[image_idx - 1].append(scaled_homo_matrix)


class MSCOCO(Dataset):
    """
        This class loads the MS-COCO dataset [1].

        Reference:
            [1] Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P. and Zitnick, C.L., 2014, September. Microsoft coco: Common objects in context. In European conference on computer vision (pp. 740-755). Springer, Cham.
    """

    def __init__(self, dataset_folder, transform, in_image_sampling_min_crop_size=0.0, in_image_sampling_likelihood=0.0, input_size=(240., 320.), max_loaded_samples=0):
        # Variables
        self.dataset_folder = dataset_folder
        self.transform = transform
        self.in_image_sampling_min_crop_size = in_image_sampling_min_crop_size
        self.in_image_sampling_likelihood = in_image_sampling_likelihood
        self.h, self.w = input_size
        self.data = []

        # Load dataset
        self._load()

        # Randomly pick k images
        if max_loaded_samples > 0:
            np.random.shuffle(self.data)
            self.data = self.data[:max_loaded_samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get image
        img = self._load_img(self.data[idx])

        #  In-image sampling: positive samples from the same image
        if self.in_image_sampling_min_crop_size > 0.0:
            pos_sam_1, pos_sam_2, neg_sam = dip.sample_pairs(img, self.in_image_sampling_min_crop_size)
        else:
            pos_sam_1 = img
            pos_sam_2 = img
            neg_sam = None

        # Apply transforms
        pos_sam_1 = self.transform(pos_sam_1)
        pos_sam_2 = self.transform(pos_sam_2)

        # Negative samples from the same image
        if (not (neg_sam is None)) and (np.random.rand() <= self.in_image_sampling_likelihood):
            has_neg_sam = True
            neg_sam_1 = self.transform(neg_sam)
            neg_sam_2 = self.transform(neg_sam)
        else:
            neg_sam_1, neg_sam_2 = torch.empty(pos_sam_1.size()), torch.empty(pos_sam_1.size())
            has_neg_sam = False

        return pos_sam_1, pos_sam_2, neg_sam_1, neg_sam_2, has_neg_sam

    def _load(self):
        files = os.listdir(self.dataset_folder)
        for i in range(len(files)):
            img_path = os.path.join(self.dataset_folder, files[i])
            if os.path.isfile(img_path):
                print(img_path)
                self.data.append(img_path)

    def _load_img(self, path):
        # Read image
        img_np = cv.cvtColor(cv.imread(path, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)

        # Scaling that keeps the aspect ratio
        h, w, c = img_np.shape
        h_s = self.h / float(h)
        w_s = self.w / float(w)
        s = max(h_s, w_s)

        # Center crop
        center_h = int(h / 2)
        center_new_h = int(int(self.h / s) / 2)
        center_w = int(w / 2)
        center_new_w = int(int(self.w / s) / 2)
        img_np = img_np[center_h - center_new_h:center_h + center_new_h, center_w - center_new_w:center_w + center_new_w]

        # Return PIL image
        return Image.fromarray(np.uint8(img_np))


class MSCOCOLocal(Dataset):
    """
        This class loads the dataset composed of local crops from MS-COCO [1] based on the keypoints detected by CorrNet.

        Reference:
            [1] Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P. and Zitnick, C.L., 2014, September. Microsoft coco: Common objects in context. In European conference on computer vision (pp. 740-755). Springer, Cham.
    """

    def __init__(self, dataset_folder, transform, max_loaded_samples=0):
        # Variables
        self.dataset_folder = dataset_folder
        self.transform = transform
        self.data = []

        # Load dataset
        self._load()

        # Randomly pick k images
        if max_loaded_samples > 0:
            np.random.shuffle(self.data)
            self.data = self.data[:max_loaded_samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # To return
        pos_sam_1 = []
        pos_sam_2 = []

        # Get image
        full_paths_to_image = self.data[idx]

        # Apply transforms
        for full_path_to_image in full_paths_to_image:
            img_np = cv.cvtColor(cv.imread(full_path_to_image, cv.IMREAD_COLOR), cv.COLOR_BGR2RGB)
            image = Image.fromarray(np.uint8(img_np))
            pos_sam_1.append(self.transform(image))
            pos_sam_2.append(self.transform(image))

        return torch.stack(pos_sam_1), torch.stack(pos_sam_2)

    def _load(self):
        folder_images = np.sort(np.array(os.listdir(self.dataset_folder)))

        # Iterate over images
        for folder_image in folder_images:
            folder_image = os.path.join(self.dataset_folder, folder_image)
            folder_keypoints = np.sort(np.array((os.listdir(folder_image))))
            print(folder_image)

            # Iterate over keypoints
            for folder_keypoint in folder_keypoints:
                folder_keypoint = os.path.join(folder_image, folder_keypoint)
                file_local_patches = np.sort(np.array((os.listdir(folder_keypoint))))

                # Iterate over local patches
                local_patches = []
                for file_local_patch in file_local_patches:
                    full_path_to_local_patch = os.path.join(folder_keypoint, file_local_patch)
                    local_patches.append(full_path_to_local_patch)
                self.data.append(local_patches)
