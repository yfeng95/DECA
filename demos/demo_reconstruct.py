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
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg

def main(args):
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    # load test images 
    testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector)

    # run DECA
    deca_cfg.model.use_tex = args.useTex
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca = DECA(config = deca_cfg, device=device)
    # for i in range(len(testdata)):
    for i in tqdm(range(len(testdata))):
        name = testdata[i]['imagename']
        images = testdata[i]['image'].to(device)[None,...]
        with torch.no_grad():
            codedict = deca.encode(images)
            opdict, visdict = deca.decode(codedict) #tensor
        if args.saveDepth or args.saveKpt or args.saveObj or args.saveMat or args.saveImages:
            os.makedirs(os.path.join(savefolder, name), exist_ok=True)
        # -- save results
        if args.saveDepth:
            depth_image = deca.render.render_depth(opdict['trans_verts']).repeat(1,3,1,1)
            visdict['depth_images'] = depth_image
            cv2.imwrite(os.path.join(savefolder, name, name + '_depth.jpg'), util.tensor2image(depth_image[0]))
        if args.saveKpt:
            np.savetxt(os.path.join(savefolder, name, name + '_kpt2d.txt'), opdict['landmarks2d'][0].cpu().numpy())
            np.savetxt(os.path.join(savefolder, name, name + '_kpt3d.txt'), opdict['landmarks3d'][0].cpu().numpy())
        if args.saveObj:
            deca.save_obj(os.path.join(savefolder, name, name + '.obj'), opdict)
        if args.saveMat:
            opdict = util.dict_tensor2npy(opdict)
            savemat(os.path.join(savefolder, name, name + '.mat'), opdict)
        if args.saveVis:
            cv2.imwrite(os.path.join(savefolder, name + '_vis.jpg'), deca.visualize(visdict))
        if args.saveImages:
            for vis_name in ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images']:
                if vis_name not in visdict.keys():
                    continue
                image  =util.tensor2image(visdict[vis_name][0])
                cv2.imwrite(os.path.join(savefolder, name, name + '_' + vis_name +'.jpg'), util.tensor2image(visdict[vis_name][0]))
            # overlap on input (face segmented) image
            wfp = os.path.join(savefolder, name, name + '_' + 'overlap' +'.jpg')
            alpha = 0.6
            deca.render.render_shape(opdict['verts'], opdict['trans_verts'])
            res = cv2.addWeighted(util.tensor2image(visdict['inputs'][0]), 1 - alpha, util.tensor2image(visdict['shape_detail_images'][0]), alpha, 0)
            cv2.imwrite(wfp, res)
            # save original image
            cv2.imwrite(os.path.join(savefolder, name, name + '_' + 'original_image' +'.jpg'), util.tensor2image(testdata[i]['original_image']))
            # get full shape image
            shape_full = torch.cat((torch.cat((visdict['shifted_shape_lt'], visdict['shifted_shape_rt']), 3),
                                    torch.cat((visdict['shifted_shape_lb'], visdict['shifted_shape_rb']), 3)),2)
            cv2.imwrite(os.path.join(savefolder, name, name + '_' + 'shape_full' +'.jpg'), util.tensor2image(shape_full[0]))
            # overlap full shape image to original image
            wfp = os.path.join(savefolder, name, name + '_' + 'original_overlap' +'.jpg')
            dst_image = util.render_overlap(shape_full, testdata[i])
            #cv2.imwrite(os.path.join(savefolder, name, name + '_' + 'shape_trans' +'.jpg'), util.tensor2image(dst_image)) #.transpose(2,0,1).float()
            alpha = 0.6
            res = cv2.addWeighted(util.tensor2image(testdata[i]['original_image']), 1 - alpha, util.tensor2image(dst_image), alpha, 0)
            cv2.imwrite(wfp, res)
    print(f'-- please check the results in {savefolder}')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-i', '--inputpath', default='TestSamples/examples', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder', default='TestSamples/examples/results', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu' )
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check decalib/detectors.py for details' )
    # rendering option
    parser.add_argument('--rasterizer_type', default='standard', type=str,
                        help='rasterizer type: pytorch3d or standard' )
    # save
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model' )
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output' )
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save 2D and 3D keypoints' )
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save depth image' )
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj, detail mesh will end with _detail.obj. \
                            Note that saving objs could be slow' )
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat' )
    parser.add_argument('--saveImages', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images' )
    main(parser.parse_args())