import os
import h5py
import argparse
import numpy as np
from tqdm import tqdm

def construct_hoi4d_part(root_dir: str, dataset_dir: str, config_dir: str):
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

    train_list = ""
    val_list = ""
    test_list = ""

    for h5file in ["train1"]:
        with h5py.File(f"{root_dir}/{h5file}.h5", "r") as f:
            for data, folder, suffix, data_type in zip(
                ["semantic", "pcd"], ["labels", "velodyne"], ["label", "bin"], [np.int32, np.float32]
            ):
                print(h5file, data)
                original_data = f[data]
                shape = original_data.shape
                assert shape[1] == 300
                for video in tqdm(range(500)):
                    video_folder = dir_path + "/{:0>3d}/{}".format(video, folder)
                    os.makedirs(video_folder)
                    for frame in range(160):
                        filename = video_folder + "/{:0>6d}.{}".format(frame, suffix)
                        frame_data = original_data[video][frame].astype(data_type)
                        frame_data.tofile(filename)
                        if data == "pcd":
                            if filename == "test": test_list += filename + "\n"
                            else:
                                if video % 5 == 0: val_list += filename + "\n"
                                else: train_list += filename + "\n"
                        
    f_train = open(f"{config_dir}/train_data.txt", "w")
    f_train.write(train_list)
    f_train.close()
    f_val = open(f"{config_dir}/val_data.txt", "w")
    f_val.write(val_list)
    f_val.close()
    f_test = open(f"{config_dir}/test_data.txt", "w")
    f_test.write(test_list)
    f_test.close()

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

    train_list = ""
    val_list = ""
    test_list = ""
    video_cnt = -1
    
    # train and validation data
    for h5file in ["train1", "train2", "train3", "train4"]:
        with h5py.File(f"{root_dir}/{h5file}.h5", "r") as f:
            for data, folder, suffix, data_type in zip(
                ["semantic", "pcd"], ["labels", "velodyne"], ["label", "bin"], [np.int32, np.float32]
            ):
                print(h5file, data)
                original_data = f[data]
                shape = original_data.shape
                for video in tqdm(range(shape[0])):
                    video_cnt += 1
                    video_folder = dir_path + "/{:0>4d}/{}".format(video_cnt, folder)
                    os.makedirs(video_folder)
                    for frame in range(shape[1]):
                        filename = video_folder + "/{:0>6d}.{}".format(frame, suffix)
                        frame_data = original_data[video][frame].astype(data_type)
                        frame_data.tofile(filename)
                        if data == "pcd":
                            if video % 5 == 0: val_list += filename + "\n"
                            else: train_list += filename + "\n"
    
    # test data 
    with h5py.File(f"{root_dir}/test.h5", "r") as f:
        data, folder, suffix, data_type = "pcd", "velodyne", "bin", np.float32
        original_data = f[data]
        shape = original_data.shape
        for video in tqdm(range(shape[0])):
            video_cnt += 1
            video_folder = dir_path + "/{:0>4d}/{}".format(video_cnt, folder)
            os.makedirs(video_folder)
            for frame in range(shape[1]):
                filename = video_folder + "/{:0>6d}.{}".format(frame, suffix)
                frame_data = original_data[video][frame].astype(data_type)
                frame_data.tofile(filename)
                test_list += filename + "\n"

    f_train = open(f"{config_dir}/train_data.txt", "w")
    f_train.write(train_list)
    f_train.close()
    f_val = open(f"{config_dir}/val_data.txt", "w")
    f_val.write(val_list)
    f_val.close()
    f_test = open(f"{config_dir}/test_data.txt", "w")
    f_test.write(test_list)
    f_test.close()


def construct_kitti(root_dir: str, config_dir: str):
    """
    Construct filelist for KITTI.
    mode: 'train' use 00-10; 'test' use 11-21.

    Args:
    root_dir: path to KITTI.
    config_dir: path to config folder where we save filelist.
    """

    train_list = ""
    val_list = ""
    test_list = ""

    path = root_dir + "/dataset/sequences/"
    videos = os.listdir(path)
    videos.sort()
    for video in videos:
        pcd_dir = path + video + "/velodyne/"
        pcd_files = os.listdir(pcd_dir)
        pcd_files.sort()
        if int(video) >= 11:
            for pcd in pcd_files:
                test_list += pcd_dir + pcd + "\n"
        elif int(video) == 8:
            for pcd in pcd_files:
                val_list += pcd_dir + pcd + "\n"
        else:
            for pcd in pcd_files:
                train_list += pcd_dir + pcd + "\n"

    f_train = open(f"{config_dir}/train_data.txt", "w")
    f_train.write(train_list)
    f_train.close()
    f_val = open(f"{config_dir}/val_data.txt", "w")
    f_val.write(val_list)
    f_val.close()
    f_test = open(f"{config_dir}/test_data.txt", "w")
    f_test.write(test_list)
    f_test.close()


def hoi4d_range(path: str, chunk_size: int = 20):
    """
    Find data range for hoi4d

    Args:
    chunk_size: size of data processed at one time. NOTE set it according to memory of your device.
    path: path to dir containing h5 files of HOI4D.
    """
    max_range = np.ones((3, 1)) * (-1e8)
    min_range = np.ones((3, 1)) * 1e8
    record = dict()

    def find_range(datas, func):
        rg = np.zeros((len(datas), 1))
        for i, data in enumerate(datas):
            rg[i] = func(data)
        return rg

    for filename in [
        "test_float32.h5",
        "train1_float32.h5",
        "train2_float32.h5",
        "train3_float32.h5",
        "train4_float32.h5",
    ]:
        with h5py.File(path + "/" + filename, "r") as f:
            print(filename)
            original_data = f["pcd"]

            shape = original_data.shape
            total_data = shape[0]
            num_iterations = total_data // chunk_size
            if total_data % chunk_size > 0:
                num_iterations += 1

            for i in tqdm(range(num_iterations)):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, total_data)
                chunk_data = np.array(original_data[start_idx:end_idx])
                datas = [chunk_data[:, :, :, i] for i in range(3)]
                max_tmp = find_range(datas, np.max)
                min_tmp = find_range(datas, np.min)
                max_range = np.max(
                    np.concatenate([max_range, max_tmp], axis=-1), axis=-1
                ).reshape((3, 1))
                min_range = np.min(
                    np.concatenate([min_range, min_tmp], axis=-1), axis=-1
                ).reshape((3, 1))
                record[filename] = [max_range, min_range]

    print(f"max: {max_range}")
    print(f"min: {min_range}")
    print(record)


if __name__ == "__main__":
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--root_dir", type=str, required=False, default="../dataset/SemanticKITTI")
    # parser.add_argument("--dataset", type=str, required=True)
    # args = parser.parse_args()
    
    root_dir = "/mnt/sdc/wangx/HOI4D/HOI4D_dataset/seg_data_h5"
    dataset_dir = "/mnt/sdc/wangx/dataset/HOI4D"
    config_dir = "/mnt/sdc/wangx/HexFormer/config/HOI4D"
    
    # hoi4d_range(root_dir)
    construct_hoi4d(root_dir, dataset_dir, config_dir)
    
    # construct_kitti(root_dir, config_dir)
