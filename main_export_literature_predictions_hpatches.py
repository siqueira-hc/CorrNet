# -*- coding: utf-8 -*-

# External modules
import numpy as np
import cv2 as cv
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os

# Internal modules
from model.dataset import HPatches
from model import dip
from model.demo_superpoint import SuperPointFrontend


# cv.setNumThreads(0)


# Params
dataset_folder = './HPatches/hpatches-sequences-release/'
experiment_path = 'Results/'

# Existing methods
# FAST, Harris, Shi-Tomasi, Random, SuperPoint
experiment_name = 'Harris'
show_keypoints = False
num_of_keypoints = 1000
top_k = 1000

print('Experiment: ', experiment_name)

# Set 0 for illumination and 1 for viewpoint changes
images_set = 1

# Initiate FAST
fast = cv.FastFeatureDetector_create()
fast.setNonmaxSuppression(1)

# SuperPoint
# sp = SuperPointFrontend(weights_path='./superpoint_v1.pth', nms_dist=3, conf_thresh=0.015, nn_thresh=0.7, cuda=True)
sp = SuperPointFrontend(weights_path='./superpoint_v1.pth', nms_dist=3, conf_thresh=0.0001, nn_thresh=0.7, cuda=True)

# Evaluate classical methods
evaluation_full_path = os.path.join(experiment_path, experiment_name, 'illumination' if images_set == 0 else 'viewpoint')

# Check singularity
if not os.path.exists(evaluation_full_path):
    os.makedirs(evaluation_full_path)

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

file_counter = -1
img_counter = 1
keypoints_stat = []
while True:
    try:
        # Get data
        ref_data = next(list_data_iter[0])
        ref_img, _, _, _, _, _ = ref_data

        # Convert to numpy
        ref_img_np = dip.tensor2numpy_image(ref_img[0].detach())
        ref_img_np_gray = cv.cvtColor(ref_img_np, cv.COLOR_RGB2GRAY)

        print('\nDetecting keypoints in reference image: ', img_counter)
        img_counter += 1

        # Load illumination images
        if (images_set == 0) and (img_counter >= 59):
            break
        # Load viewpoint images
        elif (images_set == 1) and (img_counter < 59):
            for i in range(1, 6):
                tar_data = next(list_data_iter[i])
            continue

        for i in range(1, 6):
            file_counter += 1

            # Get tar image
            tar_data = next(list_data_iter[i])
            tar_img, _, _, _, _, tar_h = tar_data

            # Convert to numpy
            tar_img_np = dip.tensor2numpy_image(tar_img[0].detach())
            tar_img_np_gray = cv.cvtColor(tar_img_np, cv.COLOR_RGB2GRAY)

            # Homography
            tar_h = tar_h[0].numpy()

            # Find keypoints
            if 'FAST' in experiment_name:
                # Reference
                t = 600
                fast.setThreshold(t)
                ref_keypoints = fast.detect(ref_img_np_gray, None)

                while len(ref_keypoints) < num_of_keypoints:
                    t -= 5
                    fast.setThreshold(t)
                    ref_keypoints = fast.detect(ref_img_np_gray, None)
                    if t < 10:
                        break

                # Target
                t = 600
                fast.setThreshold(t)
                tar_keypoints = fast.detect(tar_img_np_gray, None)

                while len(tar_keypoints) < num_of_keypoints:
                    t -= 5
                    fast.setThreshold(t)
                    tar_keypoints = fast.detect(tar_img_np_gray, None)
                    if t < 10:
                        break

                # Convert to Numpy format
                ref_keypoints_np = np.vstack([[key.pt[1], key.pt[0], key.response] for key in ref_keypoints])
                tar_keypoints_np = np.vstack([[key.pt[1], key.pt[0], key.response] for key in tar_keypoints])

                print(np.max(ref_keypoints_np, axis=0))
                print(np.max(tar_keypoints_np, axis=0))
            elif 'Harris' in experiment_name:
                # Reference
                ref_detections = cv.cornerHarris(ref_img_np_gray, 3, 3, 0.04)
                t = 0.1
                ref_detections_bool = ref_detections > (t * ref_detections.max())
                while np.sum(ref_detections_bool) < num_of_keypoints:
                    t -= 0.0005
                    ref_detections_bool = ref_detections > (t * ref_detections.max())
                ref_keypoints = []
                for x in range(ref_detections.shape[0]):
                    for y in range(ref_detections.shape[1]):
                        if ref_detections_bool[x, y]:
                            ref_keypoints.append([x, y, ref_detections[x, y]])
                ref_keypoints_np = np.vstack(ref_keypoints)

                # Target
                tar_detections = cv.cornerHarris(tar_img_np_gray, 3, 3, 0.04)
                t = 0.1
                tar_detections_bool = tar_detections > (t * tar_detections.max())
                while np.sum(tar_detections_bool) < num_of_keypoints:
                    t -= 0.0005
                    tar_detections_bool = tar_detections > (t * tar_detections.max())
                tar_keypoints = []
                for x in range(tar_detections.shape[0]):
                    for y in range(tar_detections.shape[1]):
                        if tar_detections_bool[x, y]:
                            tar_keypoints.append([x, y, tar_detections[x, y]])
                tar_keypoints_np = np.vstack(tar_keypoints)
            elif 'Shi-Tomasi' in experiment_name:
                # Reference
                ref_keypoints = cv.goodFeaturesToTrack(ref_img_np_gray, num_of_keypoints, 0.001, 3)
                ref_keypoints_np = ref_keypoints[:, 0, [1, 0]]
                ref_keypoints_np = np.hstack([ref_keypoints_np, np.ones((ref_keypoints_np.shape[0], 1))])

                # Target
                tar_keypoints = cv.goodFeaturesToTrack(tar_img_np_gray, num_of_keypoints, 0.001, 3)
                tar_keypoints_np = tar_keypoints[:, 0, [1, 0]]
                tar_keypoints_np = np.hstack([tar_keypoints_np, np.ones((tar_keypoints_np.shape[0], 1))])
            elif 'Random' in experiment_name:
                ref_keypoints_np = np.random.rand(ref_img_np_gray.shape[0], ref_img_np_gray.shape[1])
                tar_keypoints_np = np.random.rand(tar_img_np_gray.shape[0], tar_img_np_gray.shape[1])
            elif 'SuperPoint' in experiment_name:
                # points, desc, detections = sp.run(.)
                ref_point, ref_desc, ref_keypoints_np = sp.run(ref_img_np_gray.astype(np.float32)/255.)
                ref_point = ref_point.T
                ref_point = ref_point[:, [1, 0, 2]]
                ref_desc = ref_desc.T
                tar_point, tar_desc, tar_keypoints_np = sp.run(tar_img_np_gray.astype(np.float32)/255.)
                tar_point = tar_point.T
                tar_point = tar_point[:, [1, 0, 2]]
                tar_desc = tar_desc.T
            else:
                ref_keypoints_np = None
                tar_keypoints_np = None

            # Convert to 2D
            if ('Random' in experiment_name) or ('SuperPoint' in experiment_name):
                ref_grads = ref_keypoints_np
                tar_grads = tar_keypoints_np
            else:
                ref_grads = np.zeros(ref_img_np_gray.shape)
                for x, y, p in ref_keypoints_np:
                    ref_grads[int(x), int(y)] = p

                tar_grads = np.zeros(tar_img_np_gray.shape)
                for x, y, p in tar_keypoints_np:
                    tar_grads[int(x), int(y)] = p

            # NMS
            desc1, desc2 = None, None
            if 'SuperPoint' in experiment_name:
                ref_keypoints_np = ref_point
                desc1 = ref_desc
                tar_keypoints_np = tar_point
                desc2 = tar_desc
            else:
                ref_keypoints_np = dip.detect_classical(ref_grads, top_k=top_k)
                tar_keypoints_np = dip.detect_classical(tar_grads, top_k=top_k)

            # Show number of keypoints
            print('Reference keypoints: ', ref_keypoints_np.shape[0])
            print('Target keypoints: ', tar_keypoints_np.shape[0])

            # Stats
            keypoints_stat.append(ref_keypoints_np.shape[0])
            keypoints_stat.append(tar_keypoints_np.shape[0])

            # Show keypoints
            if show_keypoints:
                ref_img_np = cv.cvtColor(ref_img_np, cv.COLOR_RGB2BGR)
                dip.draw_keypoints(ref_img_np, ref_keypoints_np)
                cv.imshow('Reference', ref_img_np)

                tar_img_np = cv.cvtColor(tar_img_np, cv.COLOR_RGB2BGR)
                dip.draw_keypoints(tar_img_np, tar_keypoints_np)
                cv.imshow('Target', tar_img_np)

                cv.waitKey(0)

            # Convert to SuperPoint format
            kp1 = ref_keypoints_np[:, [1, 0, 2]]
            kp2 = tar_keypoints_np[:, [1, 0, 2]]
            if desc1 is None:
                desc1 = np.zeros((kp1.shape[0], 256))
                desc2 = np.zeros((kp2.shape[0], 256))
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

print('Mean keypoints: ', np.mean(keypoints_stat))
print('\nFiles saved at: ', evaluation_full_path)

exit(0)
