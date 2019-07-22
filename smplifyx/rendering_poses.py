from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from mesh_viewer_background import MeshBackgroundViewer
import numpy as np
import easy_configuration
import openpose_wrapper



import sys
import os

import os.path as osp

import time
import yaml
import torch

import smplx
import cv2
import trimesh
import pyrender
import numpy as np
from utils import JointMapper
from cmd_parser import parse_config

from easy_run import get_img, get_keypoints
from human_body_prior.tools.model_loader import load_vposer
from mesh_viewer import MeshViewer

def main(**args):
    input_media = args.pop('input_media')
    config = easy_configuration.configure(**args)
    device = torch.device('cpu')
    dtype=torch.float32

    body_model = config['neutral_model']
    body_model.to(device= device)
    camera = config['camera']

    vposer_ckpt = args['vposer_ckpt']
    vposer_ckpt = osp.expandvars(vposer_ckpt)
    vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
    vposer = vposer.to(device=device)
    vposer.eval()

    viewer = MeshViewer()


    body_color=(1.0, 1.0, 0.9, 1.0)
    img_np = cv2.imread(input_media)
    img = get_img(input_media)
    # Keypoints for the first person
    keypoints = get_keypoints(img_np)[[0]]

    keypoint_data = torch.tensor(keypoints, dtype=dtype)
    gt_joints = keypoint_data[:, :, :2]
    gt_joints = gt_joints.to(device=device, dtype=dtype)


    def render_embedding(vposer, pose_embedding, body_model, viewer):
        body_pose = vposer.decode(pose_embedding, output_type='aa').view(1, -1)
        body_pose.to(device=device)
        body_model_output = body_model(body_pose=body_pose)

        vertices = body_model_output.vertices.detach().cpu().numpy().squeeze()

        viewer.update_mesh(vertices,  body_model.faces)

    while True:
        pose_embedding = torch.randn([1,32],  dtype=torch.float32, device=device) * 10929292929
        render_embedding(vposer, pose_embedding, body_model, viewer)



if __name__ == "__main__":
    args = parse_config()
    main(**args)
