import os
from smplx import SMPL
import torch
import numpy as np
import pickle as pkl
import vedo
import trimesh
import time

def pkl_reader(args):
    data_dir = args.data_dir
    seq_name = args.sequence_name
    smpl_path = args.smpl_path

    smpl = SMPL(smpl_path, batch_size=1)

    data = pkl.load(open(os.path.join(data_dir, seq_name), "rb"))

    smpl_poses = data['smpl_poses']
    smpl_trans = data['root_trans']

    print("Shape of smpl_poses:")
    print(data['smpl_poses'].shape)

    print("Shape of root_trans:")
    print(data['root_trans'].shape)

    print("Shape of smpl_betas:")
    print(data['smpl_betas'].shape)

    print("jijicaoni")

    n_persons = data['meta']['n_persons']
    print("Number of persons:", n_persons)



if __name__ =='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='motions_smpl', help='path to motions_smpl folder containing .pkl files')
    parser.add_argument('--sequence_name', type=str, default=None, help='')
    parser.add_argument('--smpl_path', type=str, default="smpl/SMPL_MALE.pkl", help='')
    args = parser.parse_args()

    pkl_reader(args)