import json
import os
import numpy as np

def load_data_aist(data_dir, interval=120, move=40, rotmat=False, external_wav=None, external_wav_rate=1,
                   music_normalize=False, wav_padding=0):
    tot = 0
    music_data, dance_data = [], []
    fnames = sorted(os.listdir(data_dir))
    # print(fnames)
    # fnames = fnames[:10]  # For debug

    if ".ipynb_checkpoints" in fnames:
        fnames.remove(".ipynb_checkpoints")
    for fname in fnames:
        path = os.path.join(data_dir, fname)
        # with open(path) as f:
        #     # print(path)
        #     sample_dict = json.loads(f.read())
        #     np_music = np.array(sample_dict['music_array'])
        try:
            with open(path, 'r') as f:
                print(path)
                sample_dict = json.loads(f.read())
                np_music = np.array(sample_dict.get('music_array', []))  # 添加一个默认值，以防 music_array 不存在
                if external_wav is not None:
                    wav_path = os.path.join(external_wav, fname.split('_')[-2] + '.json')
                    # print('load from external wav!')
                    with open(wav_path) as ff:
                        sample_dict_wav = json.loads(ff.read())
                        np_music = np.array(sample_dict_wav['music_array']).astype(np.float32)

                np_dance = np.array(sample_dict['dance_array'])

                if not rotmat:
                    root = np_dance[:, :3]  # the root
                    np_dance = np_dance - np.tile(root, (1, 24))  # Calculate relative offset with respect to root
                    np_dance[:, :3] = root

                music_sample_rate = external_wav_rate if external_wav is not None else 1
                # print('music_sample_rate', music_sample_rate)
                # print(music_sample_rate)
                if interval is not None:
                    seq_len, dim = np_music.shape
                    for i in range(0, seq_len, move):
                        i_sample = i // music_sample_rate
                        interval_sample = interval // music_sample_rate

                        music_sub_seq = np_music[i_sample: i_sample + interval_sample]
                        dance_sub_seq = np_dance[i: i + interval]

                        if len(music_sub_seq) == interval_sample and len(dance_sub_seq) == interval:
                            padding_sample = wav_padding // music_sample_rate
                            # Add paddings/context of music
                            music_sub_seq_pad = np.zeros((interval_sample + padding_sample * 2, dim),
                                                         dtype=music_sub_seq.dtype)

                            if padding_sample > 0:
                                music_sub_seq_pad[padding_sample:-padding_sample] = music_sub_seq
                                start_sample = padding_sample if i_sample > padding_sample else i_sample
                                end_sample = padding_sample if i_sample + interval_sample + padding_sample < seq_len else seq_len - (
                                        i_sample + interval_sample)
                                # print(end_sample)
                                music_sub_seq_pad[padding_sample - start_sample:padding_sample] = np_music[
                                                                                                  i_sample - start_sample:i_sample]
                                if end_sample == padding_sample:
                                    music_sub_seq_pad[-padding_sample:] = np_music[
                                                                          i_sample + interval_sample:i_sample + interval_sample + end_sample]
                                else:
                                    music_sub_seq_pad[-padding_sample:-padding_sample + end_sample] = np_music[
                                                                                                      i_sample + interval_sample:i_sample + interval_sample + end_sample]
                            else:
                                music_sub_seq_pad = music_sub_seq
                            music_data.append(music_sub_seq_pad)
                            dance_data.append(dance_sub_seq)
                            tot += 1
                            # if tot > 1:
                            #     break
                else:
                    music_data.append(np_music)
                    dance_data.append(np_dance)

        except FileNotFoundError:
            print(f"File not found: {path}")
            continue  # 直接跳
        except json.JSONDecodeError as e:
            print(f"JSON decoding error in file {path}: {str(e)}")
            continue
        except Exception as e:
            print(f"An error occurred in file {path}: {str(e)}")
            continue

            # if tot > 1:
            #     break

            # tot += 1
            # if tot > 100:
            #     break
    music_np = np.stack(music_data).reshape(-1, music_data[0].shape[1])
    music_mean = music_np.mean(0)
    music_std = music_np.std(0)
    music_std[(np.abs(music_mean) < 1e-5) & (np.abs(music_std) < 1e-5)] = 1

    # music_data_norm = [ (music_sub_seq - music_mean) / (music_std + 1e-10) for music_sub_seq in music_data ]
    # print(music_np)

    if music_normalize:
        print('calculating norm mean and std')
        music_data_norm = [(music_sub_seq - music_mean) / (music_std + 1e-10) for music_sub_seq in music_data]
        with open('/mnt/lustressd/lisiyao1/dance_experiements/music_norm.json', 'w') as fff:
            sample_dict = {
                'music_mean': music_mean.tolist(),  # musics[idx+i],
                'music_std': music_std.tolist()
            }
            # print(sample_dict)
            json.dump(sample_dict, fff)
    else:
        music_data_norm = music_data

    return music_data_norm, dance_data, ['11', '22', ]
    # , [fn.replace('.json', '') for fn in fnames]