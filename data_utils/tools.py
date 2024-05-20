import os
import h5py
import argparse
import yaml
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
    video_cnt = 0
    
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
                    video_cnt = video_cnt_base + i
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


def remap(semantic: np.array, inverse: bool = False):
        """
        Remap semantic classes.

        Args:
        semantic: semantic classes to remap.
        inverse: class2num if True, num2class if False. NOTE: See KITTI config for more.
        """
        config_path = "config/kitti/semantic-kitti-all.yaml"
        cfg = yaml.safe_load(open(config_path, "r"))
        # get number of interest classes, and the label mappings
        if inverse:
            # print("Mapping xentropy to original labels")
            remapdict = cfg["learning_map_inv"]
        else:
            remapdict = cfg["learning_map"]

        # make lookup table for mapping
        maxkey = max(remapdict.keys())

        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(remapdict.keys())] = list(remapdict.values())
        return remap_lut[semantic]
    
    
def visualize(data_dir: str, log_dir: str, config_path: str, frame_num: int, predict: bool = True):
    scan_name = data_dir + "/dataset/sequences/08/velodyne/{:0>6d}.bin".format(frame_num)
    scan = np.fromfile(scan_name, dtype=np.float32).reshape(-1, 4)[:, :-1]
    if predict:
        label_name = log_dir + "/sequences/08/predictions/{:0>6d}.label".format(frame_num)
    else:
        label_name = data_dir + "/dataset/sequences/08/labels/{:0>6d}.label".format(frame_num)
    label = np.fromfile(label_name, dtype=np.int32)
    label = label.reshape((-1))
    sem_label = label & 0xFFFF  # semantic label in lower half
    
    points = scan
    labels = remap(sem_label, False)
    
    DATA = yaml.safe_load(open(config_path, 'r'))
    remapdict = DATA["learning_map_inv"]
    # make lookup table for mapping
    maxkey = max(remapdict.keys())
    # +100 hack making lut bigger just in case there are unknown labels
    remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
    remap_lut[list(remapdict.keys())] = list(remapdict.values())
    remap_lut[-1] = -1
    labels = remap_lut[labels]
    
    color = DATA["color_map"]
    color_lut = np.ones((300, 3), dtype=np.int32)
    color_lut[list(color.keys())] = list(color.values())
    color_lut[-1] = np.ones(3, dtype=np.int32) * 100
    colors = color_lut[labels]
    
    pcds = np.concatenate([points, colors], axis=1)
    np.savetxt(f'{frame_num}-pcds-label-{predict}.txt', pcds)

if __name__ == "__main__":
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--root_dir", type=str, required=False, default="../dataset/SemanticKITTI")
    # parser.add_argument("--dataset", type=str, required=True)
    # args = parser.parse_args()
    
    # root_dir = "/mnt/sdc/wangx/HOI4D/HOI4D_dataset/seg_data_h5"
    # dataset_dir = "/mnt/sdc/wangx/dataset/HOI4D"
    # config_dir = "/mnt/sdc/wangx/HexFormer/config/HOI4D"
    # hoi4d_range(root_dir)
    # construct_hoi4d(root_dir, dataset_dir, config_dir)
    
    root_dir = "/mnt/sdc/wangrh/data/SemanticKITTI"
    config_dir = "/mnt/sdc/wangx/HexFormer/config/kitti"
    config_path = "/mnt/sdc/wangx/HexFormer/config/kitti/semantic-kitti-all.yaml"
    log_dir = "/mnt/sdc/wangx/HexFormer/logs/log_MultiScan_3Dconv_4Dattention_test_kitti"
    # construct_kitti(root_dir, config_dir)
    frame_num = 3600
    visualize(root_dir, log_dir, config_path, frame_num, predict=True)
    visualize(root_dir, log_dir, config_path, frame_num, predict=False)
