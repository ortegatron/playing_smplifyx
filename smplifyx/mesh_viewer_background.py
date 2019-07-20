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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import trimesh
import cv2
import pyrender

render_out = None
def start_render(width, height):
    global render_out
    render_out = cv2.VideoWriter('construction.avi',cv2.VideoWriter_fourcc(*'XVID'), 20, (width,height))

def finish_render():
    print("Finishingf")
    render_out.release()

def create_raymond_lights():
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3,:3] = np.c_[x,y,z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes

class MeshBackgroundViewer(object):

    def __init__(self, width=1200, height=800,
                 body_color=(1.0, 1.0, 0.9, 1.0),
                 registered_keys=None,
                 background_image = None,
                 camera = None):
        super(MeshBackgroundViewer, self).__init__()

        if registered_keys is None:
            registered_keys = dict()

        import trimesh
        import pyrender

        self.mat_constructor = pyrender.MetallicRoughnessMaterial
        self.mesh_constructor = trimesh.Trimesh
        self.trimesh_to_pymesh = pyrender.Mesh.from_trimesh
        self.transf = trimesh.transformations.rotation_matrix
        self.background_image = background_image
        self.camera = camera

        self.body_color = body_color
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 1.0],
                                    ambient_light=(0.3, 0.3, 0.3))

    def set_camera(self):
        # Adds to the scene the camera in self.camera
        focal_length=5000.
        camera = self.camera
        camera_center = camera.center.detach().cpu().numpy().squeeze()
        camera_transl = camera.translation.detach().cpu().numpy().squeeze()
        # Equivalent to 180 degrees around the y-axis. Transforms the fit to
        # OpenGL compatible coordinate system.
        camera_transl[0] *= -1.0

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_transl

        camera = pyrender.camera.IntrinsicsCamera(
            fx=focal_length, fy=focal_length,
            cx=camera_center[0], cy=camera_center[1])
        self.scene.add(camera, pose=camera_pose)

    # These methods are not used anymore but left there for compatibility
    def is_active(self):
        pass

    def close_viewer(self):
        pass

    def create_mesh(self, vertices, faces, color=(0.3, 0.3, 0.3, 1.0),
                    wireframe=False):

        material = self.mat_constructor(
            metallicFactor=0.0,
            alphaMode='BLEND',
            baseColorFactor=color)

        mesh = self.mesh_constructor(vertices, faces)

        rot = self.transf(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)

        return self.trimesh_to_pymesh(mesh, material=material)

    def update_mesh(self, vertices, faces):
        # Clean scene
        for node in self.scene.get_nodes():
            self.scene.remove_node(node)

        # Place mesh
        body_mesh = self.create_mesh(
            vertices, faces, color=self.body_color)
        self.scene.add(body_mesh, name='body_mesh')

        # Create lightning
        raymond_lights = create_raymond_lights()
        for light in raymond_lights:
            light.light.intensity = 1
            self.scene.add_node(light)
        self.set_camera()

        # Render scene
        height, width , _  = self.background_image.shape
        r = pyrender.OffscreenRenderer(viewport_width=width,
                                       viewport_height=height,
                                        point_size=1.0)
        flags = pyrender.RenderFlags.ALL_WIREFRAME 
        color, _ = r.render(self.scene, flags)
        color = color.astype(np.uint8)

        valid_mask = (color[:, :] != [0.,0.,0.])[:, :]
        output_img = (color[:, :] * valid_mask +
                      (1 - valid_mask) * self.background_image * 255)
        output_img = np.uint8(output_img)
        cv2.imshow("Frame", output_img)
        key = cv2.waitKey(1)
        # Come back to [0,255] pixel depth
        render_out.write(output_img)
