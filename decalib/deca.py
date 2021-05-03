# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os, sys
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
from time import time
from skimage.io import imread
import cv2
import pickle
from .utils.renderer import SRenderY
from .models.encoders import ResnetEncoder
from .models.FLAME import FLAME, FLAMETex
from .models.decoders import Generator
from .utils import util
from .utils.rotation_converter import batch_euler2axis
from .datasets import datasets
from .utils.config import cfg
torch.backends.cudnn.benchmark = True

class DECA(object):
    def __init__(self, config=None, device='cuda'):
        if config is None:
            self.cfg = cfg
        else:
            self.cfg = config
        self.device = device
        self.image_size = self.cfg.dataset.image_size
        self.uv_size = self.cfg.model.uv_size

        self._create_model(self.cfg.model)
        self._setup_renderer(self.cfg.model)

    def _setup_renderer(self, model_cfg):
        self.render = SRenderY(self.image_size, obj_filename=model_cfg.topology_path, uv_size=model_cfg.uv_size).to(self.device)
        # face mask for rendering details
        mask = imread(model_cfg.face_eye_mask_path).astype(np.float32)/255.; mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
        np_eye_mask = F.interpolate(mask, [model_cfg.uv_size, model_cfg.uv_size]).detach().numpy()
        np_eye_mask[np_eye_mask > 0] = 1
        self.uv_face_eye_mask = torch.from_numpy(np_eye_mask).cuda()
        mask = imread(model_cfg.face_mask_path).astype(np.float32)/255.; mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
        self.uv_face_mask = F.interpolate(mask, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        # displacement correction
        fixed_dis = np.load(model_cfg.fixed_displacement_path)
        self.fixed_uv_dis = torch.tensor(fixed_dis).float().to(self.device)
        # mean texture
        mean_texture = imread(model_cfg.mean_tex_path).astype(np.float32)/255.; mean_texture = torch.from_numpy(mean_texture.transpose(2,0,1))[None,:,:,:].contiguous()
        self.mean_texture = F.interpolate(mean_texture, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        # dense mesh template, for save detail mesh
        self.dense_template = np.load(model_cfg.dense_template_path, allow_pickle=True, encoding='latin1').item()

    def _create_model(self, model_cfg):
        # set up parameters
        self.n_param = model_cfg.n_shape+model_cfg.n_tex+model_cfg.n_exp+model_cfg.n_pose+model_cfg.n_cam+model_cfg.n_light
        self.n_detail = model_cfg.n_detail
        self.n_cond = model_cfg.n_exp + 3 # exp + jaw pose
        self.num_list = [model_cfg.n_shape, model_cfg.n_tex, model_cfg.n_exp, model_cfg.n_pose, model_cfg.n_cam, model_cfg.n_light]
        self.param_dict = {i:model_cfg.get('n_' + i) for i in model_cfg.param_list}

        # encoders
        self.E_flame = ResnetEncoder(outsize=self.n_param).to(self.device) 
        self.E_detail = ResnetEncoder(outsize=self.n_detail).to(self.device)
        # decoders
        self.flame = FLAME(model_cfg).to(self.device)
        if model_cfg.use_tex:
            self.flametex = FLAMETex(model_cfg).to(self.device)
        self.D_detail = Generator(latent_dim=self.n_detail+self.n_cond, out_channels=1, out_scale=model_cfg.max_z, sample_mode = 'bilinear').to(self.device)
        # resume model
        model_path = self.cfg.pretrained_modelpath
        if os.path.exists(model_path):
            print(f'trained model found. load {model_path}')
            checkpoint = torch.load(model_path)
            self.checkpoint = checkpoint
            util.copy_state_dict(self.E_flame.state_dict(), checkpoint['E_flame'])
            util.copy_state_dict(self.E_detail.state_dict(), checkpoint['E_detail'])
            util.copy_state_dict(self.D_detail.state_dict(), checkpoint['D_detail'])
        else:
            print(f'please check model path: {model_path}')
            exit()
        # eval mode
        self.E_flame.eval()
        self.E_detail.eval()
        self.D_detail.eval()

    def decompose_code(self, code, num_dict):
        ''' Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
        '''
        code_dict = {}
        start = 0
        for key in num_dict:
            end = start+int(num_dict[key])
            code_dict[key] = code[:, start:end]
            start = end
            if key == 'light':
                code_dict[key] = code_dict[key].reshape(code_dict[key].shape[0], 9, 3)
        return code_dict

    def displacement2normal(self, uv_z, coarse_verts, coarse_normals):
        ''' Convert displacement map into detail normal map
        '''
        batch_size = uv_z.shape[0]
        uv_coarse_vertices = self.render.world2uv(coarse_verts).detach()
        uv_coarse_normals = self.render.world2uv(coarse_normals).detach()
    
        uv_z = uv_z*self.uv_face_eye_mask
        uv_detail_vertices = uv_coarse_vertices + uv_z*uv_coarse_normals + self.fixed_uv_dis[None,None,:,:]*uv_coarse_normals.detach()
        dense_vertices = uv_detail_vertices.permute(0,2,3,1).reshape([batch_size, -1, 3])
        uv_detail_normals = util.vertex_normals(dense_vertices, self.render.dense_faces.expand(batch_size, -1, -1))
        uv_detail_normals = uv_detail_normals.reshape([batch_size, uv_coarse_vertices.shape[2], uv_coarse_vertices.shape[3], 3]).permute(0,3,1,2)
        return uv_detail_normals

    def displacement2vertex(self, uv_z, coarse_verts, coarse_normals):
        ''' Convert displacement map into detail vertices
        '''
        batch_size = uv_z.shape[0]
        uv_coarse_vertices = self.render.world2uv(coarse_verts).detach()
        uv_coarse_normals = self.render.world2uv(coarse_normals).detach()
    
        uv_z = uv_z*self.uv_face_eye_mask
        uv_detail_vertices = uv_coarse_vertices + uv_z*uv_coarse_normals + self.fixed_uv_dis[None,None,:,:]*uv_coarse_normals.detach()
        dense_vertices = uv_detail_vertices.permute(0,2,3,1).reshape([batch_size, -1, 3])
        # uv_detail_normals = util.vertex_normals(dense_vertices, self.render.dense_faces.expand(batch_size, -1, -1))
        # uv_detail_normals = uv_detail_normals.reshape([batch_size, uv_coarse_vertices.shape[2], uv_coarse_vertices.shape[3], 3]).permute(0,3,1,2)
        detail_faces =  self.render.dense_faces
        return dense_vertices, detail_faces

    def visofp(self, normals):
        ''' visibility of keypoints, based on the normal direction
        '''
        normals68 = self.flame.seletec_3d68(normals)
        vis68 = (normals68[:,:,2:] < 0.1).float()
        return vis68

    @torch.no_grad()
    def encode(self, images):
        batch_size = images.shape[0]
        parameters = self.E_flame(images)
        detailcode = self.E_detail(images)
        codedict = self.decompose_code(parameters, self.param_dict)
        codedict['detail'] = detailcode
        codedict['images'] = images
        return codedict

    @torch.no_grad()
    def decode(self, codedict):
        images = codedict['images']
        batch_size = images.shape[0]
        
        ## decode
        verts, landmarks2d, landmarks3d = self.flame(shape_params=codedict['shape'], expression_params=codedict['exp'], pose_params=codedict['pose'])
        uv_z = self.D_detail(torch.cat([codedict['pose'][:,3:], codedict['exp'], codedict['detail']], dim=1))
        if self.cfg.model.use_tex:
            albedo = self.flametex(codedict['tex'])
        else:
            albedo = torch.zeros([batch_size, 3, self.uv_size, self.uv_size], device=images.device) 
        ## projection
        landmarks2d = util.batch_orth_proj(landmarks2d, codedict['cam'])[:,:,:2]; landmarks2d[:,:,1:] = -landmarks2d[:,:,1:]; landmarks2d = landmarks2d*self.image_size/2 + self.image_size/2
        landmarks3d = util.batch_orth_proj(landmarks3d, codedict['cam']); landmarks3d[:,:,1:] = -landmarks3d[:,:,1:]; landmarks3d = landmarks3d*self.image_size/2 + self.image_size/2
        trans_verts = util.batch_orth_proj(verts, codedict['cam']); trans_verts[:,:,1:] = -trans_verts[:,:,1:]
        
        ## rendering
        ops = self.render(verts, trans_verts, albedo, codedict['light'])
        uv_detail_normals = self.displacement2normal(uv_z, verts, ops['normals'])
        uv_shading = self.render.add_SHlight(uv_detail_normals, codedict['light'])
        uv_texture = albedo*uv_shading

        landmarks3d_vis = self.visofp(ops['transformed_normals'])
        landmarks3d = torch.cat([landmarks3d, landmarks3d_vis], dim=2)

        ## render shape
        shape_images = self.render.render_shape(verts, trans_verts)
        detail_normal_images = F.grid_sample(uv_detail_normals, ops['grid'], align_corners=False)*ops['alpha_images']
        shape_detail_images = self.render.render_shape(verts, trans_verts, detail_normal_images=detail_normal_images)
        
        ## extract texture
        ## TODO: current resolution 256x256, support higher resolution, and add visibility
        uv_pverts = self.render.world2uv(trans_verts)
        uv_gt = F.grid_sample(images, uv_pverts.permute(0,2,3,1)[:,:,:,:2], mode='bilinear')
        if self.cfg.model.use_tex:
            ## TODO: poisson blending should give better-looking results
            uv_texture_gt = uv_gt[:, :3, :, :] * self.uv_face_eye_mask
        else:
            uv_texture_gt = uv_gt[:, :3, :, :] * self.uv_face_eye_mask

        texture_iamge = F.grid_sample(uv_gt, ops['grid'], align_corners=False)
            
        ## output
        opdict = {
            'vertices': verts,
            'normals': ops['normals'],
            'transformed_vertices': trans_verts,
            'landmarks2d': landmarks2d,
            'landmarks3d': landmarks3d,
            'uv_detail_normals': uv_detail_normals,
            'uv_texture_gt': uv_texture_gt,
            'displacement_map': uv_z+self.fixed_uv_dis[None,None,:,:],
        }
        if self.cfg.model.use_tex:
            opdict['albedo'] = albedo
            opdict['uv_texture'] = uv_texture

        visdict = {
            'inputs': images, 
            'landmarks2d': util.tensor_vis_landmarks(images, landmarks2d, isScale=False),
            'landmarks3d': util.tensor_vis_landmarks(images, landmarks3d, isScale=False),
            'shape_images': shape_images,
            'shape_detail_images': shape_detail_images,
            'texture_iamge': texture_iamge
        }
        if self.cfg.model.use_tex:
            visdict['rendered_images'] = ops['images']
        return opdict, visdict

    def visualize(self, visdict, size=None):
        grids = {}
        if size is None:
            size = self.image_size
        for key in visdict:
            grids[key] = torchvision.utils.make_grid(F.interpolate(visdict[key], [size, size])).detach().cpu()
        grid = torch.cat(list(grids.values()), 2)
        grid_image = (grid.numpy().transpose(1,2,0).copy()*255)[:,:,[2,1,0]]
        grid_image = np.minimum(np.maximum(grid_image, 0), 255).astype(np.uint8)
        return grid_image
    
    def save_obj(self, filename, opdict):
        '''
        vertices: [nv, 3], tensor
        texture: [3, h, w], tensor
        '''
        i = 0
        vertices = opdict['vertices'][i].cpu().numpy()
        faces = self.render.faces[0].cpu().numpy()
        texture = util.tensor2image(opdict['uv_texture_gt'][i])
        uvcoords = self.render.raw_uvcoords[0].cpu().numpy()
        uvfaces = self.render.uvfaces[0].cpu().numpy()
        # save coarse mesh, with texture and normal map
        normal_map = util.tensor2image(opdict['uv_detail_normals'][i]*0.5 + 0.5)
        util.write_obj(filename, vertices, faces, 
                        texture=texture, 
                        uvcoords=uvcoords, 
                        uvfaces=uvfaces, 
                        normal_map=normal_map)
        # upsample mesh, save detailed mesh
        texture = texture[:,:,[2,1,0]]
        normals = opdict['normals'][i].cpu().numpy()
        displacement_map = opdict['displacement_map'][i].cpu().numpy().squeeze()
        dense_vertices, dense_colors, dense_faces = util.upsample_mesh(vertices, normals, faces, displacement_map, texture, self.dense_template)
        util.write_obj(filename.replace('.obj', '_detail.obj'), 
                        dense_vertices, 
                        dense_faces,
                        colors = dense_colors,
                        inverse_face_order=True)