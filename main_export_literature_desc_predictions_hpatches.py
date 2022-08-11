# -*- coding: utf-8 -*-

# External modules
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import cv2 as cv
import numpy as np

# Internal modules
from model.dataset import HPatches
from model.cornet import CorNet
from model.guided_backprop import GuidedBackpropReLUModel
from model import dip
from model.matcher import corres_matcher


# Environment variables
matplotlib.use('Agg')
# cv.setNumThreads(0)


# ORB or SIFT
method = 'SIFT'

# Params
results_folder = './Results/{}/'.format(method)
dataset_folder = './HPatches/hpatches-sequences-release/'

# Experimental params
images_set = 0

# Create result folder
evaluation_full_path = os.path.join(results_folder, 'illumination' if images_set == 0 else 'viewpoint')
if not os.path.exists(evaluation_full_path):
    os.makedirs(evaluation_full_path)

# Initiate SIFT detector
if 'SIFT' in method:
    print('Initializing SIFT...')
    detector = cv.SIFT_create(nfeatures=1000)
else:
    print('Initializing ORB...')
    detector = cv.ORB_create(nfeatures=1000)

# Transforms for evaluation
transform_eval = transforms.Compose([transforms.Resize((240, 320)), transforms.ToTensor()])

# Load data
print('Reading data...')
list_data_iter = []
for d_i in range(6):
    d = HPatches(dataset_folder, transform_eval)
    d.set_sequence_idx(d_i)
    dl = DataLoader(d, batch_size=1, shuffle=False, num_workers=4, drop_last=False)
    list_data_iter.append(iter(dl))

# Iterate over the dataset
img_counter = 1
file_counter = -1
while True:
    try:
        # Get data
        ref_data = next(list_data_iter[0])
        ref_img, _, _, _, _, _ = ref_data

        print('Sequence: ', img_counter)
        img_counter += 1

        # Load illumination images
        if (images_set == 0) and (img_counter >= 59):
            break
        # Load viewpoint images
        elif (images_set == 1) and (img_counter < 59):
            for i in range(1, 6):
                tar_data = next(list_data_iter[i])
            continue

        # Convert to numpy
        ref_img_np = dip.tensor2numpy_image(ref_img[0].detach().cpu())
        ref_img_np = cv.cvtColor(ref_img_np, cv.COLOR_RGB2BGR)
        ref_img_np_gray = cv.cvtColor(ref_img_np, cv.COLOR_BGR2GRAY)

        # Reference
        ref_keypoints, ref_descriptions = detector.detectAndCompute(ref_img_np_gray, None)

        # Iterate over target images
        for i in range(1, 6):
            # Get tar image
            tar_data = next(list_data_iter[i])
            tar_img, _, _, _, _, tar_h = tar_data

            # Convert tar image to numpy
            tar_img_np = dip.tensor2numpy_image(tar_img[0].detach())
            tar_img_np = cv.cvtColor(tar_img_np, cv.COLOR_RGB2BGR)
            tar_img_np_gray = cv.cvtColor(tar_img_np, cv.COLOR_BGR2GRAY)
            tar_h = tar_h[0].numpy()

            # Target
            tar_keypoints, tar_descriptions = detector.detectAndCompute(tar_img_np_gray, None)

            # Convert to evaluation format
            kp1 = np.array([[int(key.pt[0]), int(key.pt[1]), 1] for key in ref_keypoints]).astype(np.float32)
            desc1 = ref_descriptions.astype(np.float32)
            kp2 = np.array([[int(key.pt[0]), int(key.pt[1]), 1] for key in tar_keypoints]).astype(np.float32)
            desc2 = tar_descriptions.astype(np.float32)
            tar_h = tar_h.astype(np.float32)

            # Export detection
            file_counter += 1
            file_name = '{}.npz'.format(file_counter)
            item = {
                'image': ref_img_np,
                'prob': kp1,
                'desc': desc1,
                'warped_image': tar_img_np,
                'warped_prob': kp2,
                'warped_desc': desc2,
                'homography': tar_h,
                'matches': None
            }
            np.savez_compressed(os.path.join(evaluation_full_path, file_name), **item)
    except StopIteration:
        break

print('\nFiles saved at: ', evaluation_full_path)
exit(0)
