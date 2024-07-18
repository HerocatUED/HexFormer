import os
import h5py
import argparse
import numpy as np
from tqdm import tqdm


def check_and_init(config_dir: str):
    """
    Check whether filelist already exists and initialize filelist.

    Args:
    config_dir: path to config folder where we save filelist.
    """
    if os.path.exists(f"{config_dir}/train_data.txt"):
        assert 0, "config files already exist!"
    if os.path.exists(f"{config_dir}/val_data.txt"):
        assert 0, "config files already exist!"
    if os.path.exists(f"{config_dir}/test_data.txt"):
        assert 0, "config files already exist!"
    f_train = open(f"{config_dir}/train_data.txt", "a")
    f_val = open(f"{config_dir}/val_data.txt", "a")
    f_test = open(f"{config_dir}/test_data.txt", "a")
    return f_train, f_val, f_test


def construct_hoi4d(root_dir: str, dataset_dir: str, config_dir: str):
    """
    Construct filelist for HOI4D.
    Uncompress data from .h5 file and organize in KITTI structure.

    Args:
    root_dir: path to HOI4D seg data.
    dataset_dir: path to save uncompressed data.
    config_dir: path to config folder where we save filelist.
    """
    dir_path = dataset_dir + "/dataset/sequences"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    f_train, f_val, f_test = check_and_init(config_dir)
    video_cnt = -1

    # train and validation data
    for h5file in ["train1", "train2", "train3", "train4"]:
        with h5py.File(f"{root_dir}/{h5file}.h5", "r") as f:
            video_cnt_base = video_cnt
            for data, folder, suffix, data_type in zip(
                ["semantic", "pcd"], ["labels", "velodyne"], ["label", "bin"], [np.int32, np.float32]
            ):
                print(h5file, data)
                original_data = f[data]
                shape = original_data.shape
                for i, video in enumerate(tqdm(range(shape[0]))):
                    video_cnt = video_cnt_base + i + 1
                    video_folder = dir_path + "/{:0>4d}/{}".format(video_cnt, folder)
                    os.makedirs(video_folder)
                    for frame in range(shape[1]):
                        filename = video_folder + "/{:0>6d}.{}".format(frame, suffix)
                        frame_data = original_data[video][frame].astype(data_type)
                        frame_data.tofile(filename)
                        if data == "pcd":
                            if video % 5 == 0: f_val.write(filename + "\n")
                            else: f_train.write(filename + "\n")
    
    # test data 
    video_cnt_base = video_cnt
    with h5py.File(f"{root_dir}/test.h5", "r") as f:
        data, folder, suffix, data_type = "pcd", "velodyne", "bin", np.float32
        original_data = f[data]
        shape = original_data.shape
        for i, video in enumerate(tqdm(range(shape[0]))):
            video_cnt = video_cnt_base + i + 1
            video_folder = dir_path + "/{:0>4d}/{}".format(video_cnt, folder)
            os.makedirs(video_folder)
            for frame in range(shape[1]):
                filename = video_folder + "/{:0>6d}.{}".format(frame, suffix)
                frame_data = original_data[video][frame].astype(data_type)
                frame_data.tofile(filename)
                f_test.write(filename + "\n")

    f_train.close()
    f_val.close()
    f_test.close()


def construct_kitti(root_dir: str, config_dir: str):
    """
    Construct filelist for KITTI.
    mode: 'train' use 00-10; 'test' use 11-21.

    Args:
    root_dir: path to KITTI.
    config_dir: path to config folder where we save filelist.
    """
    f_train, f_val, f_test = check_and_init(config_dir)
    path = root_dir + "/dataset/sequences/"
    videos = os.listdir(path)
    videos.sort()
    
    for video in videos:
        pcd_dir = path + video + "/velodyne/"
        pcd_files = os.listdir(pcd_dir)
        pcd_files.sort()
        if int(video) >= 11:
            for pcd in pcd_files:
                f_test.write(pcd_dir + pcd + "\n")
        elif int(video) == 8:
            for pcd in pcd_files:
                f_val.write(pcd_dir + pcd + "\n")
        else:
            for pcd in pcd_files:
                f_train.write(pcd_dir + pcd + "\n")

    f_train.close()
    f_val.close()
    f_test.close()
    

def prepare_dataset(dataset: str, root_dir: str):
    print(f"Preparing dataset: {dataset}.\nOriginal data should be placed in {root_dir}.")
    if dataset == "hoi4d":
        dataset_dir = "dataset/HOI4D_SemSeg"
        config_dir = "configs/hoi4d"
        construct_hoi4d(root_dir, dataset_dir, config_dir)
    elif dataset == "hoi4d_action":
        dataset_dir = "dataset/HOI4D_ActSeg"
        config_dir = "configs/hoi4d"
        construct_hoi4d(root_dir, dataset_dir, config_dir)
    elif dataset == "kitti":
        config_dir = "configs/kitti"
        construct_kitti(root_dir, config_dir)
    else:
        assert 0, "dataset not supported!"
    print(f"Dataset {dataset} done.")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--root_dir", type=str, required=True)
    args = parser.parse_args()
    prepare_dataset(args.dataset, args.root_dir)

