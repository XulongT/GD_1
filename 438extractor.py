import librosa
import numpy as np
from extractor import FeatureExtractor

extractor = FeatureExtractor()

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


# 随便test一个
file_path = 'DATA_DIR/musics/0cZwsJ1u2d0_01_321_1026.wav'
features = extract_acoustic_feature_from_file(file_path)
