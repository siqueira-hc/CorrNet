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
import numpy as np
import cv2 as cv
import argparse
import torch
from PIL import Image

# Internal modules
from model.corrnet import CorrNet
from model.guided_backprop import GuidedBackpropReLUModel
from model import dip
from model import guided_grad_cam as guided_gc


def main(path_2_reference, path_2_target, path_2_corrnet, path_2_lcorrnet, device_id, batch_size, top_k):
    # Variables
    list_images = []
    list_keypoints = []
    list_keypoints_cv = []
    list_descriptions = []
    similarity_score = 0
    mask = None

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

    # Load reference image
    ref_img_np = dip.load_image(path_2_reference)
    ref_img_pil = Image.fromarray(np.uint8(ref_img_np))
    ref_img_tensor = transform(ref_img_pil).unsqueeze(0)
    list_images.append(ref_img_np)

    # Load target image
    if path_2_target is None:
        tar_img_tensor = None
    else:
        tar_img_np = dip.load_image(path_2_target)
        tar_img_pil = Image.fromarray(np.uint8(tar_img_np))
        tar_img_tensor = transform(tar_img_pil).unsqueeze(0)
        list_images.append(tar_img_np)

    # Run the CorrNet framework
    if tar_img_tensor is None:
        # Prepare input images
        x = ref_img_tensor
        x = x.to(device)
        x.requires_grad = True

        # Detect keypoints
        ref_keypoints = dip.detect_keypoints(x, guided_bp, None, top_k=top_k)
        list_keypoints.append(ref_keypoints)
    else:
        # Prepare input images
        x = torch.cat([ref_img_tensor, tar_img_tensor])
        x = x.to(device)
        x.requires_grad = True

        # Send input images to CorrNet
        h, _ = corrnet(x)

        # Compute highest-activated output neuron
        highest_activated_neuron_id = torch.argmax(torch.mul(h[0], h[1]))

        # Compute similarity score
        similarity_score = np.clip(torch.mm(h, h.t().contiguous()).detach().cpu().numpy()[0, 1], 0., 1.)

        # Detect keypoints
        tar_img_tensor = tar_img_tensor.to(device)
        tar_img_tensor.requires_grad = True
        ref_img_tensor = ref_img_tensor.to(device)
        ref_img_tensor.requires_grad = True
        list_keypoints = guided_gc.detect_keypoints(ref_img_tensor, tar_img_tensor, corrnet, guided_bp, highest_activated_neuron_id, top_k=top_k)

        # Extract descriptions
        for i in range(2):
            descriptions = dip.extract_descriptions(x[i].unsqueeze(0), list_keypoints[i], l_corrnet, device, batch_size)
            list_descriptions.append(descriptions)

        # Convert to OpenCV format
        for keypoints in list_keypoints:
            k_cv = []
            for x, y, _ in keypoints:
                k_cv.append(cv.KeyPoint(float(y), float(x), None))
            list_keypoints_cv.append(k_cv)

    # Show image
    if len(list_images) > 1:
        # Find correspondences
        correspondences = dip.find_correspondences(list_descriptions)

        # Compute homography
        homo_matrix, homo_mask = dip.compute_homography(list_keypoints_cv, correspondences)

        # Draw correspondences
        if homo_matrix is None:
            dip.draw_keypoints(list_images[0], list_keypoints[0])
            dip.draw_keypoints(list_images[1], list_keypoints[1])
            image_2_show = cv.cvtColor(np.hstack([list_images[0], list_images[1]]), cv.COLOR_RGB2BGR)
        else:
            h, w, _ = ref_img_np.shape
            mask = homo_mask.ravel().tolist()
            image_2_show_ref = cv.cvtColor(list_images[0], cv.COLOR_RGB2BGR)
            image_2_show_tar = cv.cvtColor(list_images[1], cv.COLOR_RGB2BGR)
            draw_params = dict(matchColor=(0, 255, 0), matchesMask=mask, singlePointColor=(0, 255, 0), flags=2)
            image_2_show = cv.drawMatches(image_2_show_ref, list_keypoints_cv[0], image_2_show_tar, list_keypoints_cv[1], correspondences, None, **draw_params)

        # Add text
        h, w, c = image_2_show.shape
        text_padding = 40
        image_2_show_w_text = np.zeros((h + text_padding, w, c), dtype=np.uint8)
        image_2_show_w_text[text_padding:, :, :] = image_2_show[:]
        image_2_show = image_2_show_w_text
        text = 'Sim.: {:.1f}% - # Kps. (r): {} - # Kps (t): {} - # Corrs.: {}'.format(similarity_score*100,
                                                                                                            list_keypoints[0].shape[0],
                                                                                                            list_keypoints[1].shape[0],
                                                                                                            np.sum(mask))
        cv.putText(image_2_show, text, (5, 25), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv.LINE_AA)
    else:
        image_2_show = cv.cvtColor(list_images[0], cv.COLOR_RGB2BGR)
        dip.draw_keypoints(image_2_show, list_keypoints[0], (0, 255, 0))

        # Add text
        h, w, c = image_2_show.shape
        text_padding = 40
        image_2_show_w_text = np.zeros((h + text_padding, w, c), dtype=np.uint8)
        image_2_show_w_text[text_padding:, :, :] = image_2_show[:]
        image_2_show = image_2_show_w_text
        text = '# Kps. (r): {}'.format(list_keypoints[0].shape[0])
        cv.putText(image_2_show, text, (5, 25), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv.LINE_AA)

    cv.imshow('Output', image_2_show)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Demonstration of keypoint detection and description extraction with the CorrNet framework.')
    parser.add_argument('-r', type=str, help='Reference image.', required=True)
    parser.add_argument('-t', type=str, help='Target image.')
    parser.add_argument('-c', type=str, help='Full path to CorrNet.', default='./corrnet.pth')
    parser.add_argument('-l', type=str, help='Full path to l-CorrNet.', default='./l-corrnet.pth')
    parser.add_argument('-d', type=int, help='CUDA id. The default running device is CPU.', default=-1)
    parser.add_argument('-b', type=int, help='Batch size of l-CorrNet.', default=32)
    parser.add_argument('-k', type=int, help='Top k keypoints.', default=1000)
    args = parser.parse_args()

    # Run demo
    main(args.r, args.t, args.c, args.l, args.d, args.b, args.k)
    exit(0)
