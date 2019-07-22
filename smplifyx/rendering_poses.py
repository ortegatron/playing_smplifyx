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


def main(**args):
    input_media = args.pop('input_media')
    config = easy_configuration.configure(**args)
    body_model = config['neutral_model']
    camera = config['camera']
    device = torch.device('cpu')
    body_model.to(device= device)

    vposer_ckpt = args['vposer_ckpt']
    vposer_ckpt = osp.expandvars(vposer_ckpt)
    vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
    vposer = vposer.to(device=device)
    vposer.eval()

    dtype=torch.float32

    body_color=(1.0, 1.0, 0.9, 1.0)
    img_np = cv2.imread(input_media)
    background_image=np.zeros((500,500))
    img = get_img(input_media)
    # Keypoints for the first person
    keypoints = get_keypoints(img_np)[[0]]

    keypoint_data = torch.tensor(keypoints, dtype=dtype)
    gt_joints = keypoint_data[:, :, :2]
    gt_joints = gt_joints.to(device=device, dtype=dtype)


    def render_embedding(vposer, pose_embedding, body_model):
        body_pose = vposer.decode(pose_embedding, output_type='aa').view(1, -1)
        body_pose.to(device=device)
        body_model_output = body_model(body_pose=body_pose)

        vertices = body_model_output.vertices.detach().cpu().numpy().squeeze()
        joints = body_model_output.joints.detach().cpu().numpy().squeeze()

        print('Vertices shape =', vertices.shape)
        print('Joints shape =', joints.shape)

        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
        tri_mesh = trimesh.Trimesh(vertices, body_model.faces,
                                   vertex_colors=vertex_colors)

        mesh = pyrender.Mesh.from_trimesh(tri_mesh)

        scene = pyrender.Scene()
        scene.add(mesh)
        #plot joints
        sm = trimesh.creation.uv_sphere(radius=0.005)
        sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
        tfs = np.tile(np.eye(4), (len(joints), 1, 1))
        tfs[:, :3, 3] = joints
        joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        scene.add(joints_pcl)

        pyrender.Viewer(scene, use_raymond_lighting=True)


    pose_embedding = torch.randn([1,32],  dtype=torch.float32, device=device)
    render_embedding(vposer, pose_embedding, body_model)



if __name__ == "__main__":
    args = parse_config()
    main(**args)
