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



def main(**args):
    input_media = args.pop('input_media')
    config = easy_configuration.configure(**args)


    body_color=(1.0, 1.0, 0.9, 1.0)
    background_image=np.zeros((500,500))
    model = config['neutral_model']
    camera=config['camera']

    device = torch.device('cpu')
    print(model)
    # body_pose_prior=config['body_pose_prior']

    # body_mean_pose = body_pose_prior.get_mean().detach().cpu()
    # body_model.reset_params(body_pose=body_mean_pose)

    # model = body_model(return_verts=True, body_pose=body_mean_pose)
    betas = torch.randn([1, 10], dtype=torch.float32, device=device)
    expression = torch.randn([1, 10], dtype=torch.float32, device=device)

    body_pose = torch.randn([1,smplx.SMPLX.NUM_BODY_JOINTS*3],  dtype=torch.float32, device=device)
    left_hand_pose = torch.randn([1,model.num_pca_comps],  dtype=torch.float32, device=device)
    right_hand_pose = torch.randn([1,model.num_pca_comps],  dtype=torch.float32, device=device)

    model.to(device)
    output = model(betas=betas, expression=expression,
                body_pose = body_pose,
                left_hand_pose = left_hand_pose,
                right_hand_pose = right_hand_pose,
                   return_verts=True)

    vertices = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()

    print('Vertices shape =', vertices.shape)
    print('Joints shape =', joints.shape)

    vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
    tri_mesh = trimesh.Trimesh(vertices, model.faces,
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


if __name__ == "__main__":
    args = parse_config()
    main(**args)
