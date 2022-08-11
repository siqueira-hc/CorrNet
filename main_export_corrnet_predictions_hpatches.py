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
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import argparse
import torch
import os

# Internal modules
from model.guided_backprop import GuidedBackpropReLUModel
from model import guided_grad_cam as guided_gc
from model.dataset import HPatches
from model.corrnet import CorrNet
from model import dip


def main(path_2_dataset_folder, images_set_id, path_2_corrnet, path_2_lcorrnet, device_id, batch_size, top_k, run_single, extract_desc):
    # Variables
    counter_prediction_file = -1
    counter_sequence_id = 1

    # Device
    if device_id < 0:
        print('Demo will run on CPU.')
        device = torch.device('cpu')
    else:
        print('Demo will run on CUDA: {}'.format(device_id))
        device = torch.device('cuda:{}'.format(device_id))

    # Load CorNet
    corrnet = CorrNet(feature_dim=512)
    corrnet.load_state_dict(torch.load(path_2_corrnet, map_location=device))
    corrnet = corrnet.to(device)
    corrnet.eval()

    # Load CorNet
    l_corrnet = CorrNet(feature_dim=512)
    l_corrnet.load_state_dict(torch.load(path_2_lcorrnet, map_location=device))
    l_corrnet = l_corrnet.to(device)
    l_corrnet.eval()

    # Initialize guided backpropagation
    guided_bp = GuidedBackpropReLUModel(corrnet, device)

    # Transforms for evaluation
    transform = transforms.Compose([transforms.Resize((240, 320)), transforms.ToTensor()])

    # Check singularity
    evaluation_full_path = './corrnet_on_hpatches/illumination/' if images_set_id == 0 else './corrnet_on_hpatches/viewpoint/'
    if not os.path.exists(evaluation_full_path):
        os.makedirs(evaluation_full_path)

    # Load the dataset
    list_data_iter = []
    for d_i in range(6):
        print('Loading the HPatches dataset... Sequence {} of {}'.format(d_i + 1, 6))
        d = HPatches(path_2_dataset_folder, transform)
        d.set_sequence_idx(d_i)
        dl = DataLoader(d)
        list_data_iter.append(iter(dl))

    # Iterate over sequences
    while True:
        try:
            # Get data
            ref_data = next(list_data_iter[0])
            ref_img, _ = ref_data
            counter_sequence_id += 1

            # Load illumination images
            if (images_set_id == 0) and (counter_sequence_id >= 59):
                break
            # Load viewpoint images
            elif (images_set_id == 1) and (counter_sequence_id < 59):
                for i in range(1, 6):
                    next(list_data_iter[i])
                continue

            print('\nProcessing sequence: ', counter_sequence_id - 1)

            # Convert reference image to numpy
            ref_img_np = dip.tensor2numpy_image(ref_img[0].detach())

            # Iterate over target images
            for i in range(1, 6):
                counter_prediction_file += 1

                # Get tar image
                tar_data = next(list_data_iter[i])
                tar_img, tar_h = tar_data

                # Convert tar image to numpy
                tar_img_np = dip.tensor2numpy_image(tar_img[0].detach())
                tar_h = tar_h[0].numpy()

                # Send ref and tar image to CorNet
                input_x = torch.cat([ref_img, tar_img])
                input_x = input_x.to(device)

                # Single or correlate reference and target images
                list_keypoints = []
                if run_single:
                    # Iterate over reference and target images
                    for x_i in input_x:
                        # Prepare data
                        x_i = x_i.unsqueeze(0)
                        x_i.requires_grad = True

                        # Detect keypoints
                        keypoints = dip.detect_keypoints(x_i, guided_bp, None, top_k=top_k)
                        list_keypoints.append(keypoints)
                else:
                    # Send input images to CorrNet
                    output_h, output_z = corrnet(input_x)

                    # Compute highest-activated output neuron
                    output_targets = torch.mul(output_h[0], output_h[1])
                    output_idx = torch.argmax(output_targets)

                    # Similarity score
                    similarity_scores = torch.mm(output_h, output_h.t().contiguous()).detach().cpu().numpy()
                    print('Similarity: {:.1f}%'.format(similarity_scores[0, 1] * 100))

                    # Send to device
                    x_1 = input_x[0].unsqueeze(0)
                    x_1.requires_grad = True
                    x_2 = input_x[1].unsqueeze(0)
                    x_2.requires_grad = True

                    # Detect keypoints
                    list_keypoints = guided_gc.detect_keypoints(x_1, x_2, corrnet, guided_bp, output_idx, top_k=top_k)

                print('Keypoints in the reference image: ', list_keypoints[0].shape[0])
                print('Keypoints in the target image: ', list_keypoints[1].shape[0])

                # Extract descriptions
                list_descriptions = []
                if extract_desc:
                    for input_idx in range(2):
                        descriptions = dip.extract_descriptions(input_x[input_idx].unsqueeze(0), list_keypoints[input_idx], l_corrnet, device, batch_size)
                        list_descriptions.append(descriptions)
                else:
                    list_descriptions.append(np.zeros((list_keypoints[0].shape[0], 256), dtype=np.float32))
                    list_descriptions.append(np.zeros((list_keypoints[1].shape[0], 256), dtype=np.float32))

                # Convert to float
                kp1 = list_keypoints[0].astype(np.float32)
                kp2 = list_keypoints[1].astype(np.float32)
                tar_h = tar_h.astype(np.float32)

                # Convert to SuperPoint's evaluation format
                kp1 = kp1[:, [1, 0, 2]]
                kp2 = kp2[:, [1, 0, 2]]

                # Descriptions
                desc1 = list_descriptions[0]
                desc2 = list_descriptions[1]

                # Export prediction based on SuperPoint's evaluation format
                file_name = '{}.npz'.format(counter_prediction_file)
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


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Demonstration of keypoint detection and description extraction with the CorrNet framework.')
    parser.add_argument('-f', type=str, help='Full path to the HPatches dataset folder (e.g., /.../hpatches-sequences-release/).', required=True)
    parser.add_argument('-s', type=int, help='Image set: 0 - Illumination / 1 - Viewpoint.', required=True)
    parser.add_argument('-c', type=str, help='Full path to CorrNet.', default='./corrnet.pth')
    parser.add_argument('-l', type=str, help='Full path to l-CorrNet.', default='./l-corrnet.pth')
    parser.add_argument('-d', type=int, help='CUDA id. The default running device is CPU.', default=-1)
    parser.add_argument('-b', type=int, help='Batch size of l-CorrNet.', default=32)
    parser.add_argument('-k', type=int, help='Top k keypoints.', default=1000)
    parser.add_argument('--single', help='True to detect keypoints w.r.t. a single image.', action='store_true')
    parser.add_argument('--desc', help='True to extract descriptions.', action='store_true')

    args = parser.parse_args()

    # Run script
    main(args.f, args.s, args.c, args.l, args.d, args.b, args.k, args.single, args.desc)
    exit(0)
