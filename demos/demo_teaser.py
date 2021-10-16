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
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
import imageio
from skimage.transform import rescale
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.rotation_converter import batch_euler2axis, deg2rad
from decalib.utils.config import cfg as deca_cfg

def main(args):
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    # load test images 
    testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector)
    expdata = datasets.TestData(args.exp_path, iscrop=args.iscrop, face_detector=args.detector)
    # DECA
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca = DECA(config=deca_cfg, device=device)

    visdict_list_list = []
    for i in range(len(testdata)):
        name = testdata[i]['imagename']
        images = testdata[i]['image'].to(device)[None,...]
        with torch.no_grad():
            codedict = deca.encode(images)
            opdict, visdict = deca.decode(codedict) #tensor
        ### show shape with different views and expressions
        visdict_list = []
        max_yaw = 30
        yaw_list = list(range(0,max_yaw,5)) + list(range(max_yaw,-max_yaw,-5)) + list(range(-max_yaw,0,5))
        for k in yaw_list: #jaw angle from -50 to 50
            ## yaw angle
            euler_pose = torch.randn((1, 3))
            euler_pose[:,1] = k#torch.rand((self.batch_size))*160 - 80
            euler_pose[:,0] = 0#(torch.rand((self.batch_size))*60 - 30)*(2./euler_pose[:,1].abs())
            euler_pose[:,2] = 0#(torch.rand((self.batch_size))*60 - 30)*(2./euler_pose[:,1].abs())
            global_pose = batch_euler2axis(deg2rad(euler_pose[:,:3].cuda())) 
            codedict['pose'][:,:3] = global_pose
            codedict['cam'][:,:] = 0.
            codedict['cam'][:,0] = 8
            _, visdict_view = deca.decode(codedict)   
            visdict = {x:visdict[x] for x in ['inputs', 'shape_detail_images']}         
            visdict['pose'] = visdict_view['shape_detail_images']
            visdict_list.append(visdict)

        euler_pose = torch.zeros((1, 3))
        global_pose = batch_euler2axis(deg2rad(euler_pose[:,:3].cuda())) 
        codedict['pose'][:,:3] = global_pose
        for (i,k) in enumerate(range(0,31,2)): #jaw angle from -50 to 50        
            # expression: jaw pose
            euler_pose = torch.randn((1, 3))
            euler_pose[:,0] = k#torch.rand((self.batch_size))*160 - 80
            euler_pose[:,1] = 0#(torch.rand((self.batch_size))*60 - 30)*(2./euler_pose[:,1].abs())
            euler_pose[:,2] = 0#(torch.rand((self.batch_size))*60 - 30)*(2./euler_pose[:,1].abs())
            jaw_pose = batch_euler2axis(deg2rad(euler_pose[:,:3].cuda())) 
            codedict['pose'][:,3:] = jaw_pose
            _, visdict_view = deca.decode(codedict)     
            visdict_list[i]['exp'] = visdict_view['shape_detail_images']
            count = i

        for (i,k) in enumerate(range(len(expdata))): #jaw angle from -50 to 50        
            # expression: jaw pose
            exp_images = expdata[i]['image'].to(device)[None,...]
            exp_codedict = deca.encode(exp_images)
            # transfer exp code
            codedict['pose'][:,3:] = exp_codedict['pose'][:,3:]
            codedict['exp'] = exp_codedict['exp']
            _, exp_visdict = deca.decode(codedict)
            visdict_list[i+count]['exp'] = exp_visdict['shape_detail_images']

        visdict_list_list.append(visdict_list)
    
    ### write gif
    writer = imageio.get_writer(os.path.join(savefolder, 'teaser.gif'), mode='I')
    for i in range(len(yaw_list)):
        grid_image_list = []
        for j in range(len(testdata)):
            grid_image = deca.visualize(visdict_list_list[j][i])
            grid_image_list.append(grid_image)
        grid_image_all = np.concatenate(grid_image_list, 0)
        grid_image_all = rescale(grid_image_all, 0.6, multichannel=True) # resize for showing in github
        writer.append_data(grid_image_all[:,:,[2,1,0]])

    print(f'-- please check the teaser figure in {savefolder}')

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-i', '--inputpath', default='TestSamples/teaser', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-e', '--exp_path', default='TestSamples/exp', type=str, 
                        help='path to expression')
    parser.add_argument('-s', '--savefolder', default='TestSamples/teaser/results', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu' )
    # rendering option
    parser.add_argument('--rasterizer_type', default='standard', type=str,
                        help='rasterizer type: pytorch3d or standard' )
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check detectos.py for details' )

    main(parser.parse_args())