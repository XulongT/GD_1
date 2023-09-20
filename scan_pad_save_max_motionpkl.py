import os
import numpy as np
import pickle as pkl
import random

def scan_pkl_files_for_max_persons(data_dir):
    max_persons = 0
    # 记录已查找过的前缀
    checked_set = set()

    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith('.pkl'):
            # 获取前缀
            prefix = filename.split("_")[0]
            if prefix not in checked_set:
                with open(os.path.join(data_dir, filename), "rb") as f:
                    data = pkl.load(f)
                    n_persons = data['meta']['n_persons']
                    if n_persons > max_persons:
                        max_persons = n_persons
                checked_set.add(prefix)

    return max_persons



def pad_and_save_pkl_files(data_dir, output_dir, max_persons):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(data_dir):
        if filename.endswith('.pkl'):
            with open(os.path.join(data_dir, filename), "rb") as f:
                data = pkl.load(f)

                # 补0
                data['smpl_poses'] = np.pad(data['smpl_poses'],
                                            [(0, max_persons - data['smpl_poses'].shape[0]), (0, 0), (0, 0)])
                data['root_trans'] = np.pad(data['root_trans'],
                                            [(0, max_persons - data['root_trans'].shape[0]), (0, 0), (0, 0)])
                data['smpl_betas'] = np.pad(data['smpl_betas'],
                                            [(0, max_persons - data['smpl_betas'].shape[0]), (0, 0), (0, 0)])

                with open(os.path.join(output_dir, filename), "wb") as f_out:
                    pkl.dump(data, f_out)

def test(data_dir, output_dir, sample_size=5):
    #随机测试5个
    all_files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
    sampled_files = random.sample(all_files, sample_size)

    max_persons = scan_pkl_files_for_max_persons(data_dir)
    print(f"MAX = {max_persons}" )

    for filename in sampled_files:
        with open(os.path.join(data_dir, filename), "rb") as f:
            data = pkl.load(f)

            data['smpl_poses'] = np.pad(data['smpl_poses'], [(0, max_persons - data['smpl_poses'].shape[0]), (0, 0), (0, 0)])
            data['root_trans'] = np.pad(data['root_trans'], [(0, max_persons - data['root_trans'].shape[0]), (0, 0), (0, 0)])
            data['smpl_betas'] = np.pad(data['smpl_betas'], [(0, max_persons - data['smpl_betas'].shape[0]), (0, 0), (0, 0)])

            with open(os.path.join(output_dir, filename), "wb") as f_out:
                pkl.dump(data, f_out)

    print(f"Processed {sample_size} files for testing!")


if __name__ == "__main__":
    data_dir = 'DATA_DIR/motions_smpl'
    output_dir = 'DATA_DIR/motions_smpl_max'

    max_persons = scan_pkl_files_for_max_persons(data_dir)
    pad_and_save_pkl_files(data_dir, output_dir, max_persons)
    # test(data_dir, output_dir, sample_size=5)