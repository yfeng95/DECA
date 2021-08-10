import os, sys
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob

from . import detectors

def build_dataloader(config, is_train=True):
    data_list = []
    if 'vox1' in config.training_data:
        data_list.append(VoxelDataset(K=config.K, image_size=config.image_size, scale=[config.scale_min, config.scale_max], n_train=config.n_train, isSingle=config.isSingle))
    if 'vox2' in config.training_data:
        data_list.append(VoxelDataset(dataname='vox2', K=config.K, image_size=config.image_size, scale=[config.scale_min, config.scale_max], n_train=config.n_train, isSingle=config.isSingle))
    if 'vggface2' in config.training_data:
        data_list.append(VGGFace2Dataset(K=config.K, image_size=config.image_size, scale=[config.scale_min, config.scale_max], trans_scale=config.trans_scale, isSingle=config.isSingle))
    if 'vggface2hq' in config.training_data:
        data_list.append(VGGFace2HQDataset(K=config.K, image_size=config.image_size, scale=[config.scale_min, config.scale_max], trans_scale=config.trans_scale, isSingle=config.isSingle))
    if 'ethnicity' in config.training_data:
        data_list.append(EthnicityDataset(K=config.K, image_size=config.image_size, scale=[config.scale_min, config.scale_max], trans_scale=config.trans_scale, isSingle=config.isSingle))
    if 'coco' in config.training_data:
        data_list.append(COCODataset(image_size=config.image_size, scale=[config.scale_min, config.scale_max], trans_scale=config.trans_scale))
    if 'celebahq' in config.training_data:
        data_list.append(CelebAHQDataset(image_size=config.image_size, scale=[config.scale_min, config.scale_max], trans_scale=config.trans_scale))
    if 'now_eval' in config.training_data:
        data_list.append(NoWVal())
    if 'aflw2000' in config.training_data:
        data_list.append(AFLW2000())
    train_dataset = ConcatDataset(data_list)
    if is_train:
        drop_last = True
        shuffle = True
    else:
        drop_last = False
        shuffle = False
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=shuffle,
                            num_workers=config.num_workers,
                            pin_memory=True,
                            drop_last = drop_last)
    # print('---- data length: ', len(train_dataset))
    return train_dataset, train_loader

'''
images and keypoints: nomalized to [-1,1]
'''
class VoxelDataset(Dataset):
    def __init__(self, K, image_size, scale, trans_scale = 0, dataname='vox2', n_train=100000, isTemporal=False, isEval=False, isSingle=False):
        self.K = K
        self.image_size = image_size
        if dataname == 'vox1':
            self.kpt_suffix = '.txt'
            self.imagefolder = '/ps/project/face2d3d/VoxCeleb/vox1/dev/images_cropped'
            self.kptfolder = '/ps/scratch/yfeng/Data/VoxCeleb/vox1/landmark_2d'

            self.face_dict = {}
            for person_id in sorted(os.listdir(self.kptfolder)):
                for video_id in os.listdir(os.path.join(self.kptfolder, person_id)):   
                    for face_id in os.listdir(os.path.join(self.kptfolder, person_id, video_id)):
                        if 'txt' in face_id:
                            continue
                        key = person_id + '/' + video_id + '/' + face_id
                        # if key not in self.face_dict.keys():
                        #     self.face_dict[key] = []
                        name_list = os.listdir(os.path.join(self.kptfolder, person_id, video_id, face_id))
                        name_list = [name.split['.'][0] for name in name_list]
                        if len(name_list)<self.K:
                            continue
                        self.face_dict[key] = sorted(name_list)

        elif dataname == 'vox2':
            # clean version: filter out images with bad lanmark labels, may lack extreme pose example
            self.kpt_suffix = '.npy'
            self.imagefolder = '/ps/scratch/face2d3d/VoxCeleb/vox2/dev/images_cropped_full_height'
            self.kptfolder = '/ps/scratch/face2d3d/vox2_best_clips_annotated_torch7'
            self.segfolder = '/ps/scratch/face2d3d/texture_in_the_wild_code/vox2_best_clips_cropped_frames_seg/test_crop_size_400_batch/'

            cleanlist_path = '/ps/scratch/face2d3d/texture_in_the_wild_code/VGGFace2_cleaning_codes/vox2_best_clips_info_max_normal_50_images_loadinglist.npy'
            cleanlist = np.load(cleanlist_path, allow_pickle=True)
            self.face_dict = {}
            for line in cleanlist:
                person_id, video_id, face_id, name = line.split('/')
                key = person_id + '/' + video_id + '/' + face_id
                if key not in self.face_dict.keys():
                    self.face_dict[key] = []
                else:
                    self.face_dict[key].append(name)
            # filter face
            keys = list(self.face_dict.keys())
            for key in keys:
                if len(self.face_dict[key]) < self.K:
                    del self.face_dict[key]

        self.face_list = list(self.face_dict.keys())
        n_train = n_train if n_train < len(self.face_list) else len(self.face_list)
        self.face_list = list(self.face_dict.keys())[:n_train]
        if isEval:
            self.face_list = list(self.face_dict.keys())[:n_train][-100:]
        self.isTemporal = isTemporal
        self.scale = scale #[scale_min, scale_max]
        self.trans_scale = trans_scale #[dx, dy]
        self.isSingle = isSingle
        if isSingle:
            self.K = 1

    def __len__(self):
        return len(self.face_list)
    
    def __getitem__(self, idx):
        key = self.face_list[idx]    
        person_id, video_id, face_id = key.split('/')
        name_list = self.face_dict[key]
        ind = np.random.randint(low=0, high=len(name_list))

        images_list = []; kpt_list = []; fullname_list = []; mask_list = []
        if self.isTemporal:
            random_start = np.random.randint(low=0, high=len(name_list)-self.K)
            sample_list = range(random_start, random_start + self.K)
        else:
            sample_list = np.array((np.random.randint(low=0, high=len(name_list), size=self.K)))

        for i in sample_list:
            name = name_list[i]
            image_path = (os.path.join(self.imagefolder, person_id, video_id, face_id, name + '.png'))
            kpt_path = (os.path.join(self.kptfolder, person_id, video_id, face_id, name + self.kpt_suffix))
            seg_path = (os.path.join(self.segfolder, person_id, video_id, face_id, name + '.npy'))
                                            
            image = imread(image_path)/255.
            kpt = np.load(kpt_path)[:,:2]
            mask = self.load_mask(seg_path, image.shape[0], image.shape[1])

            ### crop information
            tform = self.crop(image, kpt)
            ## crop 
            cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
            cropped_mask = warp(mask, tform.inverse, output_shape=(self.image_size, self.image_size))
            cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)

            # normalized kpt
            cropped_kpt[:,:2] = cropped_kpt[:,:2]/self.image_size * 2  - 1

            images_list.append(cropped_image.transpose(2,0,1))
            kpt_list.append(cropped_kpt)
            mask_list.append(cropped_mask)

        ###
        images_array = torch.from_numpy(np.array(images_list)).type(dtype = torch.float32) #K,224,224,3
        kpt_array = torch.from_numpy(np.array(kpt_list)).type(dtype = torch.float32) #K,224,224,3
        mask_array = torch.from_numpy(np.array(mask_list)).type(dtype = torch.float32) #K,224,224,3

        if self.isSingle:
            images_array = images_array.squeeze()
            kpt_array = kpt_array.squeeze()
            mask_array = mask_array.squeeze()
                    
        data_dict = {
            'image': images_array,
            'landmark': kpt_array,
            'mask': mask_array
        }
        
        return data_dict

    def crop(self, image, kpt):
        left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
        top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])

        h, w, _ = image.shape
        old_size = (right - left + bottom - top)/2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])#+ old_size*0.1])
        # translate center
        trans_scale = (np.random.rand(2)*2 -1) * self.trans_scale
        center = center + trans_scale*old_size # 0.5

        scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]
        size = int(old_size*scale)

        # crop image
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        DST_PTS = np.array([[0,0], [0,self.image_size - 1], [self.image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        
        # cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
        # # change kpt accordingly
        # cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)
        return tform

    def load_mask(self, maskpath, h, w):
        # print(maskpath)
        if os.path.isfile(maskpath):
            vis_parsing_anno = np.load(maskpath)
            # atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            #     'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
            mask = np.zeros_like(vis_parsing_anno)
            # for i in range(1, 16):
            mask[vis_parsing_anno>0.5] = 1.
        else:
            mask = np.ones((h, w))
        return mask


class COCODataset(Dataset):
    def __init__(self, image_size, scale, trans_scale = 0, isEval=False):
        '''
        # 53877 faces
        K must be less than 6
        '''
        self.image_size = image_size
        self.imagefolder = '/ps/scratch/yfeng/Data/COCO/raw/train2017'
        self.kptfolder = '/ps/scratch/yfeng/Data/COCO/face/train2017_kpt'

        self.kptpath_list = os.listdir(self.kptfolder)
        self.scale = scale #[scale_min, scale_max]
        self.trans_scale = trans_scale # 0.5?

    def __len__(self):
        return len(self.kptpath_list)

    def __getitem__(self, idx):
        while(100):
            kptname = self.kptpath_list[idx]
            name = kptname.split('_')[0]
            image_path = os.path.join(self.imagefolder, name + '.jpg')  
            kpt_path = os.path.join(self.kptfolder, kptname)
                                            
            kpt = np.loadtxt(kpt_path)[:,:2]
            left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
            top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
            if (right - left) < 10 or (bottom - top) < 10:
                idx = np.random.randint(low=0, high=len(self.kptpath_list))
                continue

            image = imread(image_path)/255.
            if len(image.shape) < 3:
                image = np.tile(image[:,:,None], 3)
            ### crop information
            tform = self.crop(image, kpt)
            ## crop 
            cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
            cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)

            # normalized kpt
            cropped_kpt[:,:2] = cropped_kpt[:,:2]/self.image_size * 2  - 1

            ###
            images_array = torch.from_numpy(cropped_image.transpose(2,0,1)).type(dtype = torch.float32) #224,224,3
            kpt_array = torch.from_numpy(cropped_kpt).type(dtype = torch.float32) #224,224,3
                        
            data_dict = {
                'image': images_array*2. - 1,
                'landmark': kpt_array,
                # 'mask': mask_array
            }
            
            return data_dict
        
    def crop(self, image, kpt):
        left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
        top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])

        h, w, _ = image.shape
        old_size = (right - left + bottom - top)/2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])#+ old_size*0.1])
        trans_scale = (np.random.rand(2)*2 -1) * self.trans_scale
        center = center + trans_scale*old_size # 0.5
        
        scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]

        size = int(old_size*scale)

        # crop image
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        DST_PTS = np.array([[0,0], [0,self.image_size - 1], [self.image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        
        # cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
        # # change kpt accordingly
        # cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)
        return tform
    
    def load_mask(self, maskpath, h, w):
        # print(maskpath)
        if os.path.isfile(maskpath):
            vis_parsing_anno = np.load(maskpath)
            # atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            #     'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
            mask = np.zeros_like(vis_parsing_anno)
            # for i in range(1, 16):
            mask[vis_parsing_anno>0.5] = 1.
        else:
            mask = np.ones((h, w))
        return mask


class CelebAHQDataset(Dataset):
    def __init__(self, image_size, scale, trans_scale = 0, isEval=False):
        '''
        # 53877 faces
        K must be less than 6
        '''
        self.image_size = image_size
        self.imagefolder = '/ps/project/face2d3d/faceHQ_100K/celebA-HQ/celebahq_resized_256'
        self.kptfolder = '/ps/project/face2d3d/faceHQ_100K/celebA-HQ/celebahq_resized_256_torch'

        self.kptpath_list = os.listdir(self.kptfolder)
        self.scale = scale #[scale_min, scale_max]
        self.trans_scale = trans_scale # 0.5?

    def __len__(self):
        return len(self.kptpath_list)

    def __getitem__(self, idx):
        while(100):
            kptname = self.kptpath_list[idx]
            name = kptname.split('.')[0]
            image_path = os.path.join(self.imagefolder, name + '.png')  
            kpt_path = os.path.join(self.kptfolder, kptname)    
            kpt = np.load(kpt_path, allow_pickle=True)
            if len(kpt.shape) != 2:
                idx = np.random.randint(low=0, high=len(self.kptpath_list))
                continue
            # print(kpt_path, kpt.shape)
            # kpt = kpt[:,:2]

            image = imread(image_path)/255.
            if len(image.shape) < 3:
                image = np.tile(image[:,:,None], 3)
            ### crop information
            tform = self.crop(image, kpt)
            ## crop 
            cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
            cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)

            # normalized kpt
            cropped_kpt[:,:2] = cropped_kpt[:,:2]/self.image_size * 2  - 1

            ###
            images_array = torch.from_numpy(cropped_image.transpose(2,0,1)).type(dtype = torch.float32) #224,224,3
            kpt_array = torch.from_numpy(cropped_kpt).type(dtype = torch.float32) #224,224,3
                        
            data_dict = {
                'image': images_array,
                'landmark': kpt_array,
                # 'mask': mask_array
            }
            
            return data_dict
        
    def crop(self, image, kpt):
        left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
        top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])

        h, w, _ = image.shape
        old_size = (right - left + bottom - top)/2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])#+ old_size*0.1])
        trans_scale = (np.random.rand(2)*2 -1) * self.trans_scale
        center = center + trans_scale*old_size # 0.5
        
        scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]

        size = int(old_size*scale)

        # crop image
        # src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        src_pts = np.array([[0,0], [0,h - 1], [w - 1, 0]])
        DST_PTS = np.array([[0,0], [0,self.image_size - 1], [self.image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        
        # cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
        # # change kpt accordingly
        # cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)
        return tform
    
    def load_mask(self, maskpath, h, w):
        # print(maskpath)
        if os.path.isfile(maskpath):
            vis_parsing_anno = np.load(maskpath)
            # atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            #     'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
            mask = np.zeros_like(vis_parsing_anno)
            # for i in range(1, 16):
            mask[vis_parsing_anno>0.5] = 1.
        else:
            mask = np.ones((h, w))
        return mask


########################## testing
def video2sequence(video_path):
    videofolder = video_path.split('.')[0]        
    os.makedirs(videofolder, exist_ok=True)
    video_name = video_path.split('/')[-1].split('.')[0]
    # import ipdb; ipdb.set_trace()
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    imagepath_list = []
    while success:
        imagepath = '{}/{}_frame{:04d}.jpg'.format(videofolder, video_name, count)
        cv2.imwrite(imagepath, image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1
        imagepath_list.append(imagepath)
    print('video frames are stored in {}'.format(videofolder))
    return imagepath_list
    
class TestData(Dataset):
    def __init__(self, testpath, iscrop=True, crop_size=224, scale=1.25, face_detector='fan', face_detector_model=None):
        '''
            testpath: folder, imagepath_list, image path, video path
        '''
        if isinstance(testpath, list):
            self.imagepath_list = testpath
        elif os.path.isdir(testpath): 
            self.imagepath_list = glob(testpath + '/*.jpg') +  glob(testpath + '/*.png') + glob(testpath + '/*.bmp')
        elif os.path.isfile(testpath) and (testpath[-3:] in ['jpg', 'png', 'bmp']):
            self.imagepath_list = [testpath]
        elif os.path.isfile(testpath) and (testpath[-3:] in ['mp4', 'csv', 'vid', 'ebm']):
            self.imagepath_list = video2sequence(testpath)
        else:
            print('please check the input path')
            exit()
            
        print('total {} images'.format(len(self.imagepath_list)))
        self.imagepath_list = sorted(self.imagepath_list)
        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size
        if face_detector == 'dlib':
            self.face_detector = detectors.Dlib(model_path=face_detector_model)
        elif face_detector == 'fan':
            self.face_detector = detectors.FAN()
        else:
            print('no detector is used')

    def __len__(self):
        return len(self.imagepath_list)

    def __getitem__(self, index):
        imagepath = self.imagepath_list[index]
        imagename = imagepath.split('/')[-1].split('.')[0]

        image = np.array(imread(imagepath))
        if len(image.shape) == 2:
            image = image[:,:,None].repeat(1,1,3)

        h, w, _ = image.shape
        if self.iscrop:
            if max(h, w) > 1000:
                print('image is too large, resize ', imagepath) # dlib detector will be very slow if the input image size is too large
                scale_factor = 1000/max(h,w)
                image_small = rescale(image, scale_factor, preserve_range=True, multichannel=True)
                # print(image.shape)
                # print(image_small.shape)
                # exit()
                detected_faces = self.face_detector.run(image_small.astype(np.uint8))                
            else:
                detected_faces = self.face_detector.run(image.astype(np.uint8))

            if detected_faces is None:
                print('no face detected! run original image')
                left = 0; right = h-1; top=0; bottom=w-1
            else:
                # d = detected_faces[0].rect ## only use the first detected face (assume that each input image only contains one face)
                # left = d.left(); right = d.right(); top = d.top(); bottom = d.bottom()
                kpt = detected_faces[0]
                left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
                top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
                if max(h, w) > 1000:
                    scale_factor = 1./scale_factor
                    left = left*scale_factor; right = right*scale_factor; top = top*scale_factor; bottom = bottom*scale_factor
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])#+ old_size*0.1])
            size = int(old_size*self.scale)
            src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        else:
            src_pts = np.array([[0, 0], [0, h-1], [w-1, 0]])
        
        DST_PTS = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        
        image = image/255.

        dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        dst_image = dst_image.transpose(2,0,1)
        return {'image': torch.tensor(dst_image).float(),
                'imagename': imagename,
                'tform': tform,
                'original_image': torch.tensor(image.transpose(2,0,1)).float(),
                }
    
class EvalData(Dataset):
    def __init__(self, testpath, kptfolder, iscrop=True, crop_size=224, scale=1.25, face_detector='fan', face_detector_model=None):
        '''
            testpath: folder, imagepath_list, image path, video path
        '''
        if isinstance(testpath, list):
            self.imagepath_list = testpath
        elif os.path.isdir(testpath): 
            self.imagepath_list = glob(testpath + '/*.jpg') +  glob(testpath + '/*.png')
        elif os.path.isfile(testpath) and (testpath[-3:] in ['jpg', 'png']):
            self.imagepath_list = [testpath]
        elif os.path.isfile(testpath) and (testpath[-3:] in ['mp4', 'csv', 'vid', 'ebm']):
            self.imagepath_list = video2sequence(testpath)
        else:
            print('please check the input path')
            exit()
            
        # print('total {} images'.format(len(self.imagepath_list)))
        self.imagepath_list = sorted(self.imagepath_list)
        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size
        if face_detector == 'dlib':
            self.face_detector = detectors.Dlib(model_path=face_detector_model)
        elif face_detector == 'fan':
            self.face_detector = detectors.FAN()
        else:
            print('no detector is used')
        self.kptfolder = kptfolder

    def __len__(self):
        return len(self.imagepath_list)

    def __getitem__(self, index):
        imagepath = self.imagepath_list[index]
        imagename = imagepath.split('/')[-1].split('.')[0]

        image = imread(imagepath)[:,:,:3]

        h, w, _ = image.shape
        if self.iscrop:
            kptpath = os.path.join(self.kptfolder, imagename+'.npy')
            kpt = np.load(kptpath)
            left = np.min(kpt[:,0]); right = np.max(kpt[:,0])
            top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
            old_size = (right - left + bottom - top)/2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])#+ old_size*0.1])
            size = int(old_size*self.scale)
            src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        else:
            src_pts = np.array([[0, 0], [0, h-1], [w-1, 0]])
        
        DST_PTS = np.array([[0,0], [0,self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        
        image = image/255.

        dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        dst_image = dst_image.transpose(2,0,1)
        return {'image': torch.tensor(dst_image).float(),
                'imagename': imagename,
                'tform': tform,
                'original_image': torch.tensor(image.transpose(2,0,1)).float(),
                }
    
