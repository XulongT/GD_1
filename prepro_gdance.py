import os
import sys
import json
import random
import argparse
import essentia
import essentia.streaming
from essentia.standard import *
import librosa
import numpy as np
from extractor import FeatureExtractor
from smplx import SMPL
import torch

parser = argparse.ArgumentParser()
# parser.add_argument('--input_video_dir', type=str, default='aist_plusplus_final/all_musics')
# parser.add_argument('--input_annotation_dir', type=str, default='aist_plusplus_final')
parser.add_argument('--smpl_dir', type=str, default='smpl')

# parser.add_argument('--train_dir', type=str, default='data/aistpp_train_wav')
# parser.add_argument('--test_dir', type=str, default='data/aistpp_test_full_wav')

parser.add_argument('--split_train_file', type=str, default='DATA_DIR/train_split_sequence_names.txt')
parser.add_argument('--split_test_file', type=str, default='DATA_DIR/test_split_sequence_names.txt')
parser.add_argument('--split_val_file', type=str, default='DATA_DIR/val_split_sequence_names.txt')

parser.add_argument('--sampling_rate', type=int, default=15360*2)
args = parser.parse_args()

extractor = FeatureExtractor()

# if not os.path.exists(args.train_dir):
#     os.mkdir(args.train_dir)
# if not os.path.exists(args.test_dir):
#     os.mkdir(args.test_dir)
#
# split_train_file = args.split_train_file
# split_test_file = args.split_test_file
# split_val_file = args.split_val_file
#
# def make_music_dance_set(video_dir):
#     print('---------- Extract features from raw audio ----------')
#     musics = []
#     dances = []
#     fnames = []
#     train = []
#     test = []
#     train_file = open(split_train_file, 'r')
#     for fname in train_file.readlines():
#         train.append(fname.strip())
#     train_file.close()
#
#     test_file = open(split_test_file, 'r')
#     for fname in test_file.readlines():
#         test.append(fname.strip())
#     test_file.close()
#
#     test_file = open(split_val_file, 'r')
#     for fname in test_file.readlines():
#         test.append(fname.strip())
#     test_file.close()
#
#     all_names = train + test
#todo: finish this


def extract_acoustic_feature_from_file(file_path):
    # 加载音频文件
    audio, sr = librosa.load(file_path, sr=None)

    return extract_acoustic_feature(audio, sr)


def extract_acoustic_feature(audio, sr):
    melspe_db = extractor.get_melspectrogram(audio, sr)

    mfcc = extractor.get_mfcc(melspe_db)
    mfcc_delta = extractor.get_mfcc_delta(mfcc)

    audio_harmonic, audio_percussive = extractor.get_hpss(audio)

    chroma_cqt = extractor.get_chroma_cqt(audio_harmonic, sr, octave=7 if sr == 15360 * 2 else 5)

    onset_env = extractor.get_onset_strength(audio_percussive, sr)
    tempogram = extractor.get_tempogram(onset_env, sr)
    onset_beat, _ = extractor.get_onset_beat(onset_env, sr)

    onset_env = onset_env.reshape(1, -1)

    feature = np.concatenate([
        mfcc,  # 20
        mfcc_delta,  # 20
        chroma_cqt,  # 12
        onset_env,  # 1
        onset_beat,  # 1
        tempogram
    ], axis=0)

    feature = feature.transpose(1, 0)
    print(f'提取出音频维度 : {feature.shape}')

    return feature

def align(musics, dances):
    print('---------- Align the frames of music and dance ----------')
    assert len(musics) == len(dances), \
        'the number of audios should be equal to that of videos'
    new_musics=[]
    new_dances=[]
    for i in range(len(musics)):
        min_seq_len = min(len(musics[i]), len(dances[i]))
        print(f'music -> {np.array(musics[i]).shape}, ' +
              f'dance -> {np.array(dances[i]).shape}, ' +
              f'min_seq_len -> {min_seq_len}')

        new_musics.append([musics[i][j] for j in range(min_seq_len)])
        new_dances.append([dances[i][j] for j in range(min_seq_len)])

    return new_musics, new_dances, musics