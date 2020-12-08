import numpy as np
import torch

class FAN(object):
    def __init__(self):
        import face_alignment
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    def run(self, image):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        out = self.model.get_landmarks(image)
        if out is None:
            return [0]
        else:
            kpt = out[0].squeeze()
            left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
            top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
            bbox = [left,top, right, bottom]
            return bbox, 'kpt68'

class MTCNN(object):
    def __init__(self, device = 'cpu'):
        '''
        https://github.com/timesler/facenet-pytorch/blob/master/examples/infer.ipynb
        '''
        from facenet_pytorch import MTCNN as mtcnn
        self.device = device
        self.model = mtcnn(keep_all=True)
    def run(self, input):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box
        '''
        out = self.model.detect(input[None,...])
        if out[0][0] is None:
            return [0]
        else:
            bbox = out[0][0].squeeze()
            return bbox, 'bbox'



