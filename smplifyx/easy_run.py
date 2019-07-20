# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems and the Max Planck Institute for Biological
# Cybernetics. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de


# python3 smplifyx/main.py --config cfg_files/fit_smplx.yaml \
#     --data_folder go   \
#     --output_folder out  \
#     --visualize="True/False" \
#     --model_folder /home/marcelo/hands/smpl/models_smplx_v1_0/models/  \
#     --vposer_ckpt /home/marcelo/hands/smpl/vposer_v1_0
#     --visualize True




from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os

import os.path as osp

import time
import yaml
import torch

import smplx
import cv2
import numpy as np
from utils import JointMapper
from cmd_parser import parse_config
from data_parser import create_dataset
from fit_single_frame import fit_single_frame

from camera import create_camera
from prior import create_prior

torch.backends.cudnn.enabled = False

import easy_configuration
import openpose_wrapper

import mesh_viewer_background

def get_img(img_path):
    img = cv2.imread(img_path).astype(np.float32)[:, :, ::-1] / 255.0
    return img

from data_parser import Keypoints
def get_keypoints(image_np, use_hands=True, use_face=True,
                   use_face_contour=False):
    op_datum = openpose_wrapper.detect_keypoints(image_np)

    keypoints = []
    for idx, body_pose in enumerate(op_datum.poseKeypoints):
        body_keypoints = np.array(body_pose, dtype=np.float32)
        body_keypoints = body_keypoints.reshape([-1, 3])
        if use_hands:
            left_hand_keyp = np.array(
                op_datum.handKeypoints[0][idx],
                dtype=np.float32).reshape([-1, 3])
            right_hand_keyp = np.array(
                op_datum.handKeypoints[1][idx],
                dtype=np.float32).reshape([-1, 3])

            body_keypoints = np.concatenate(
                [body_keypoints, left_hand_keyp, right_hand_keyp], axis=0)
        if use_face:
            # TODO: Make parameters, 17 is the offset for the eye brows,
            # etc. 51 is the total number of FLAME compatible landmarks
            face_keypoints = np.array(
                op_datum.faceKeypoints[idx],
                dtype=np.float32).reshape([-1, 3])[17: 17 + 51, :]

            contour_keyps = np.array(
                [], dtype=body_keypoints.dtype).reshape(0, 3)
            if use_face_contour:
                contour_keyps = np.array(
                    op_datum.faceKeypoints[idx],
                    dtype=np.float32).reshape([-1, 3])[:17, :]

            body_keypoints = np.concatenate(
                [body_keypoints, face_keypoints, contour_keyps], axis=0)

        keypoints.append(body_keypoints)
    keyp_tuple =  Keypoints(keypoints=keypoints, gender_pd=[],  gender_gt=[])
    if len(keyp_tuple.keypoints) < 1:
        return {}
    keypoints = np.stack(keyp_tuple.keypoints)
    return keypoints

def main(**args):
    input_media = args.pop('input_media')
    config = easy_configuration.configure(**args)
    # supongo que es una imagen
    img_np = cv2.imread(input_media)
    img = get_img(input_media)
    keypoints = get_keypoints(img_np)

    mesh_viewer_background.start_render(img.shape[1],img.shape[0])
    for person_id in range(keypoints.shape[0]):
        # assumes neutral gender.
        body_model = config['neutral_model']

        out_img_fn = 'temp.png'
        fit_single_frame(img, keypoints[[person_id]],
                         body_model=body_model,
                         camera=config['camera'],
                         joint_weights=config['joint_weights'],
                         shape_prior=config['shape_prior'],
                         expr_prior=config['expr_prior'],
                         body_pose_prior=config['body_pose_prior'],
                         left_hand_prior=config['left_hand_prior'],
                         right_hand_prior=config['right_hand_prior'],
                         jaw_prior=config['jaw_prior'],
                         angle_prior=config['angle_prior'],
                         out_img_fn= out_img_fn,
                         **args)
        # fit_single_frame saves partial detection result to 'temp.png',
        # here we load that image again so as to render the next detection on top of that image again
        img = cv2.imread(out_img_fn).astype(np.float32)[:, :, ::-1] / 255.0
    # show the frame to our screen
    cv2.imshow("Frame", img)
    # cv2.imshow("Frame", mask)
    key = cv2.waitKey(0)
    mesh_viewer_background.finish_render()
if __name__ == "__main__":
    args = parse_config()
    main(**args)
