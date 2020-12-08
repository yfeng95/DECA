import os, sys
import cv2
import numpy as np
from time import time
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets 
from decalib.utils import util

def main(args):
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    # load test images 
    testdata = datasets.TestData(args.image_path, iscrop=args.iscrop, face_detector=args.detector)
    expdata = datasets.TestData(args.exp_path, iscrop=args.iscrop, face_detector=args.detector)

    # run DECA
    i = 0
    deca = DECA(device=device)
    name = testdata[i]['imagename']
    savepath = '{}/{}.jpg'.format(savefolder, name)
    images = testdata[i]['image'].to(device)[None,...]
    codedict = deca.encode(images)
    _, visdict = deca.decode(codedict)
    visdict = {x:visdict[x] for x in ['inputs', 'shape_detail_images']}   

    # -- expression transfer
    # exp code from image
    exp_images = expdata[i]['image'].to(device)[None,...]
    exp_codedict = deca.encode(exp_images)
    # transfer exp code
    codedict['pose'][:,3:] = exp_codedict['pose'][:,3:]
    codedict['exp'] = exp_codedict['exp']
    _, exp_visdict = deca.decode(codedict)
    visdict['transferred_shape'] = exp_visdict['shape_detail_images']
    cv2.imwrite(os.path.join(savefolder, name + '_animation.jpg'), deca.visualize(visdict))

    print(f'-- please check the results in {savefolder}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-i', '--image_path', default='TestSamples/examples/IMG_0392_inputs.jpg', type=str,
                        help='path to input image')
    parser.add_argument('-e', '--exp_path', default='TestSamples/exp/4.jpg', type=str, 
                        help='path to expression')
    parser.add_argument('-s', '--savefolder', default='TestSamples/animation_results', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default='cuda', type=str,
                        help='set device, cpu for using cpu' )
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped' )
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check detectos.py for details' )
    main(parser.parse_args())
