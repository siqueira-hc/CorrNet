# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Evaluation script adapted from [1, 2, 3].

    Reference:
        [1] SuperPoint in Tensorflow. Available at: https://github.com/rpautrat/SuperPoint. [Accessed 23 Mar. 2021].
        [2] SuperPoint from MagicLeap. Available at: https://github.com/magicleap/SuperPointPretrainedNetwork. [Accessed 23 Mar. 2021].
        [3] SuperPoint in PyTorch. Available at: https://github.com/eric-yyjau/pytorch-superpoint. [Accessed 23 Mar. 2021].
"""

__author__ = "..."
__email__ = "..."
__license__ = "..."
__version__ = "1.0"

# External modules
from tqdm import tqdm
import numpy as np
import argparse
import os
import cv2 as cv

# Internal modules
from model import dip


def select_k_best_kp_desc(points, descriptions, k):
    sorted_prob = points
    sorted_desc = descriptions
    if points.shape[1] > 2:
        sorted_prob = points[points[:, 2].argsort(), :2]
        sorted_desc = descriptions[points[:, 2].argsort(), :]
        start = min(k, points.shape[0])
        sorted_prob = sorted_prob[-start:, :]
        sorted_desc = sorted_desc[-start:, :]
    return sorted_prob, sorted_desc


def warp_keypoints(keypoints, homo_m):
    num_points = keypoints.shape[0]
    homogeneous_points = np.concatenate([keypoints, np.ones((num_points, 1))], axis=1)
    warped_points = np.dot(homogeneous_points, np.transpose(homo_m))
    return warped_points[:, :2] / warped_points[:, 2:]


def select_k_best(points, k):
    sorted_prob = points
    if points.shape[1] > 2:
        sorted_prob = points[points[:, 2].argsort(), :2]
        start = min(k, points.shape[0])
        sorted_prob = sorted_prob[-start:, :]
    return sorted_prob


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def find_files_with_ext(directory, extension='.npz', if_int=True):
    list_of_files = []

    if extension == '.npz':
        for l_i in os.listdir(directory):
            if l_i.endswith(extension):
                list_of_files.append(l_i)
    if if_int:
        list_of_files = [e for e in list_of_files if isfloat(e[:-4])]
    return list_of_files


def compute_repeatability(data, keep_k_points, distance_thresh):
    localization_err = -1
    repeatability = []
    N1s = []
    N2s = []
    H = data['homography']
    keypoints = data['prob']
    warped_keypoints = data['warped_prob']

    # Warp the original keypoints with the true homography
    true_warped_keypoints = keypoints
    true_warped_keypoints[:, :2] = warp_keypoints(keypoints[:, :2], H)

    # Keep only the keep_k_points best predictions
    warped_keypoints = select_k_best(warped_keypoints, keep_k_points)
    true_warped_keypoints = select_k_best(true_warped_keypoints, keep_k_points)

    # Compute the repeatability
    N1 = true_warped_keypoints.shape[0]
    N2 = warped_keypoints.shape[0]
    N1s.append(N1)
    N2s.append(N2)
    true_warped_keypoints = np.expand_dims(true_warped_keypoints, 1)
    warped_keypoints = np.expand_dims(warped_keypoints, 0)
    norm = np.linalg.norm(true_warped_keypoints - warped_keypoints, ord=None, axis=2)
    count1 = 0
    count2 = 0
    local_err1, local_err2 = None, None
    if N2 != 0:
        min1 = np.min(norm, axis=1)
        count1 = np.sum(min1 <= distance_thresh)
        local_err1 = min1[min1 <= distance_thresh]
    if N1 != 0:
        min2 = np.min(norm, axis=0)
        count2 = np.sum(min2 <= distance_thresh)
        local_err2 = min2[min2 <= distance_thresh]
    if N1 + N2 > 0:
        repeatability = (count1 + count2) / (N1 + N2)
    if count1 + count2 > 0:
        localization_err = 0
        if local_err1 is not None:
            localization_err += (local_err1.sum()) / (count1 + count2)
        if local_err2 is not None:
            localization_err += (local_err2.sum()) / (count1 + count2)
    else:
        repeatability = 0

    return repeatability, localization_err


def compute_homography(data, keep_k_points, correctness_thresh):
    # GT homography
    real_H = data['homography']

    # Reference keypoints and descriptions
    keypoints = data['prob']
    desc = data['desc']

    # Target keypoints and descriptions
    warped_keypoints = data['warped_prob']
    warped_desc = data['warped_desc']

    # Top K
    keypoints, desc = select_k_best_kp_desc(keypoints, desc, keep_k_points)
    keypoints = keypoints[:, [1, 0]]
    warped_keypoints, warped_desc = select_k_best_kp_desc(warped_keypoints, warped_desc, keep_k_points)
    warped_keypoints = warped_keypoints[:, [1, 0]]

    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    cv2_matches = bf.match(desc, warped_desc)
    matches_idx = np.array([m.queryIdx for m in cv2_matches])
    m_keypoints = keypoints[matches_idx, :]
    matches_idx = np.array([m.trainIdx for m in cv2_matches])
    m_warped_keypoints = warped_keypoints[matches_idx, :]

    # Estimate the homography between the matches using RANSAC
    estimated_homo_matrix, mask = cv.findHomography(m_keypoints[:, [1, 0]], m_warped_keypoints[:, [1, 0]], cv.RANSAC)

    # Compute correctness
    shape = data['image'].shape[:2]
    corners = np.array([[0, 0, 1],
                        [0, shape[0] - 1, 1],
                        [shape[1] - 1, 0, 1],
                        [shape[1] - 1, shape[0] - 1, 1]])
    real_warped_corners = np.dot(corners, np.transpose(real_H))
    real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
    warped_corners = np.dot(corners, np.transpose(estimated_homo_matrix))
    warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
    mean_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
    correctness = mean_dist <= correctness_thresh
    return correctness


def main(compute_rep, compute_homo, output_img, images_set_id, top_k):
    # Variables
    homography_thresh = [1, 3, 5]
    rep_thd = 3
    localization_err = []
    repeatability = []
    correctness = []
    counter_files = -1
    repeatability_ave = 0
    localization_err_m = 0
    correctness_ave = 0
    path_rep = None
    path_corr = None

    # Load prediction files
    path_2_predictions = './corrnet_on_hpatches/illumination/' if images_set_id == 0 else './corrnet_on_hpatches/viewpoint/'
    files = find_files_with_ext(path_2_predictions)
    files.sort(key=lambda x: int(x[:-4]))

    # Create sub-directories
    if output_img:
        path_corr = path_2_predictions + '/correspondence'
        os.makedirs(path_corr, exist_ok=True)
        path_rep = path_2_predictions + '/repeatability'
        os.makedirs(path_rep, exist_ok=True)

    # Iterate over files
    for f in tqdm(files):
        # Get data
        counter_files += 1
        data = np.load(path_2_predictions + '/' + f)
        print('\nFile: {}'.format(counter_files))

        # Compute repeatability
        if compute_rep:
            rep, local_err = compute_repeatability(data, top_k, rep_thd)
            repeatability.append(rep)
            localization_err.append(local_err)
            print('Repeatability: {:.1f}%'.format(np.round(rep * 100, 1)))
            print('Local error: {:.2f}'.format(np.round(local_err, 2)))

            # Save images
            if output_img:
                ref_img_np = data['image'].copy()
                tar_img_np = data['warped_image'].copy()
                ref_keypoints = data['prob'][:, [1, 0, 2]]
                tar_keypoints = data['warped_prob'][:, [1, 0, 2]]
                dip.draw_keypoints(ref_img_np, ref_keypoints)
                dip.draw_keypoints(tar_img_np, tar_keypoints)
                image_2_show = cv.cvtColor(np.hstack([ref_img_np, tar_img_np]), cv.COLOR_RGB2BGR)
                h, w, c = image_2_show.shape
                text_padding = 40
                image_2_show_w_text = np.zeros((h + text_padding, w, c), dtype=np.uint8)
                image_2_show_w_text[text_padding:, :, :] = image_2_show[:]
                image_2_show = image_2_show_w_text
                text = 'Repeatability: {:.1f}%'.format(np.round(rep * 100, 1))
                cv.putText(image_2_show, text, (5, 25), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv.LINE_AA)
                cv.imwrite(os.path.join(path_rep, '{}.jpg'.format(counter_files)), image_2_show)

        # Compute homography estimation accuracy
        if compute_homo:
            homo_correctness = compute_homography(data, top_k, homography_thresh)
            correctness.append(homo_correctness)
            print('Homo. est. ({}): {}'.format(str(homography_thresh), str(homo_correctness)))

            # Save images
            if output_img:
                list_keypoints_cv = []
                ref_img_np = data['image'].copy()
                tar_img_np = data['warped_image'].copy()
                ref_keypoints = data['prob'][:, [1, 0, 2]]
                tar_keypoints = data['warped_prob'][:, [1, 0, 2]]
                for keypoints in [ref_keypoints, tar_keypoints]:
                    k_cv = []
                    for x, y, _ in keypoints:
                        k_cv.append(cv.KeyPoint(float(y), float(x), None))
                    list_keypoints_cv.append(k_cv)
                correspondences = dip.find_correspondences([data['desc'], data['warped_desc']])
                _, homo_mask = dip.compute_homography(list_keypoints_cv, correspondences)
                h, w, _ = ref_img_np.shape
                mask = homo_mask.ravel().tolist()
                image_2_show_ref = cv.cvtColor(ref_img_np, cv.COLOR_RGB2BGR)
                image_2_show_tar = cv.cvtColor(tar_img_np, cv.COLOR_RGB2BGR)
                draw_params = dict(matchColor=(0, 255, 0), matchesMask=mask, singlePointColor=(0, 255, 0), flags=2)
                image_2_show = cv.drawMatches(image_2_show_ref, list_keypoints_cv[0], image_2_show_tar, list_keypoints_cv[1], correspondences, None, **draw_params)
                h, w, c = image_2_show.shape
                text_padding = 40
                image_2_show_w_text = np.zeros((h + text_padding, w, c), dtype=np.uint8)
                image_2_show_w_text[text_padding:, :, :] = image_2_show[:]
                image_2_show = image_2_show_w_text
                text = 'Homography est. acc ({}): {}'.format(str(homography_thresh), str(homo_correctness))
                cv.putText(image_2_show, text, (5, 25), cv.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv.LINE_AA)
                cv.imwrite(os.path.join(path_corr, '{}.jpg'.format(counter_files)), image_2_show)

    print('\n----------------------------------------------------')
    print('RESULTS: ')

    if compute_rep:
        repeatability_ave = np.array(repeatability).mean()
        localization_err_m = np.array(localization_err).mean()
        print('Mean repeatability: {:.1f}%'.format(np.round(repeatability_ave * 100, 1)))
        print('Mean loc. error over {}: {:.2f}'.format(len(localization_err), np.round(localization_err_m, 2)))

    if compute_homo:
        correctness_ave = np.array(correctness).mean(axis=0)
        print('Mean homography est. acc. (t = {}, {}, and {}): {:.1f}%, {:.1f}%, {:.1f}%'.format(homography_thresh[0],
                                                                                                 homography_thresh[1],
                                                                                                 homography_thresh[2],
                                                                                                 np.round(correctness_ave[0] * 100, 1),
                                                                                                 np.round(correctness_ave[1] * 100, 1),
                                                                                                 np.round(correctness_ave[2] * 100, 1)))

    print('----------------------------------------------------\n')

    # Save results at...
    path_2_results_file = path_2_predictions + './result.txt'
    with open(path_2_results_file, 'a') as file_results:
        if compute_rep:
            file_results.write('Repeatability:\n')
            file_results.write('Repeatability threshold: {}\n'.format(rep_thd))
            file_results.write('Mean repeatability: {}\n'.format(str(repeatability_ave)))
            file_results.write('Mean loc. error: {}\n'.format(str(localization_err_m)))

        if compute_homo:
            file_results.write('Homography estimation:\n')
            file_results.write('Homography threshold: {}\n'.format(str(homography_thresh)))
            file_results.write('Mean homography estimation accuracy: {}\n'.format(str(correctness_ave)))


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Demonstration of keypoint detection and description extraction with the CorrNet framework.')
    parser.add_argument('-r', help='Compute repeatability.', action='store_true')
    parser.add_argument('-e', help='Compute homography estimation.', action='store_true')
    parser.add_argument('-o', help='Save prediction images.', action='store_true')
    parser.add_argument('-s', type=int, help='Image set: 0 - Illumination / 1 - Viewpoint.', required=True)
    parser.add_argument('-k', type=int, help='Top k keypoints.', default=1000)
    args = parser.parse_args()

    # Compute results
    main(args.r, args.e, args.o, args.s, args.k)
    exit(0)
