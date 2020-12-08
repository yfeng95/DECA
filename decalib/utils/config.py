'''
Default config for DECA
'''
from yacs.config import CfgNode as CN
import argparse
import yaml
import os

cfg = CN()

abs_deca_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
cfg.deca_dir = abs_deca_dir
cfg.device = 'cuda'
cfg.device_id = '0'

cfg.pretrained_modelpath = os.path.join(cfg.deca_dir, 'data', 'deca_model.tar')

# ---------------------------------------------------------------------------- #
# Options for Face model
# ---------------------------------------------------------------------------- #
cfg.model = CN()
cfg.model.topology_path = os.path.join(cfg.deca_dir, 'data', 'head_template.obj')
cfg.model.fixed_displacement_path = os.path.join(cfg.deca_dir, 'data', 'fixed_displacement_256.npy')
cfg.model.flame_model_path = os.path.join(cfg.deca_dir, 'data', 'generic_model.pkl') 
cfg.model.flame_lmk_embedding_path = os.path.join(cfg.deca_dir, 'data', 'landmark_embedding.npy') 
cfg.model.face_mask_path = os.path.join(cfg.deca_dir, 'data', 'uv_face_mask.png') 
cfg.model.face_eye_mask_path = os.path.join(cfg.deca_dir, 'data', 'uv_face_eye_mask.png') 
cfg.model.tex_path = os.path.join(cfg.deca_dir, 'data', 'FLAME_albedo_from_BFM.npz') 
cfg.model.tex_type = 'BFM' # BFM, FLAME, albedoMM
cfg.model.uv_size = 256
cfg.model.param_list = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
cfg.model.n_shape = 100
cfg.model.n_tex = 50
cfg.model.n_exp = 50
cfg.model.n_cam = 3
cfg.model.n_pose = 6
cfg.model.n_light = 27
cfg.model.use_tex = False
cfg.model.jaw_type = 'aa' # default use axis angle, another option: euler

## details
cfg.model.n_detail = 128
cfg.model.max_z = 0.01

# ---------------------------------------------------------------------------- #
# Options for Dataset
# ---------------------------------------------------------------------------- #
cfg.dataset = CN()
cfg.dataset.batch_size = 24
cfg.dataset.num_workers = 2
cfg.dataset.image_size = 224

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()

def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')

    args = parser.parse_args()
    print(args, end='\n\n')

    cfg = get_cfg_defaults()
    if args.cfg is not None:
        cfg_file = args.cfg
        cfg = update_cfg(cfg, args.cfg)
        cfg.cfg_file = cfg_file

    return cfg
