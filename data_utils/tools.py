import os
import h5py
import yaml
import argparse
import numpy as np
from tqdm import tqdm


def hoi4d_range(path: str, chunk_size: int = 20):
    """
    Find data range for hoi4d

    Args:
    chunk_size: size of data processed at one time. NOTE set it according to memory of your device.
    path: path to dir containing h5 files of HOI4D.
    """
    
    def find_range(datas, func):
        rg = np.zeros((len(datas), 1))
        for i, data in enumerate(datas):
            rg[i] = func(data)
        return rg
    
    def find_class():
        class_ids = np.array([-1])
        for filename in [
            "train1_float32.h5",
            "train2_float32.h5",
            "train3_float32.h5",
            "train4_float32.h5",
        ]:
            with h5py.File(path + "/" + filename, "r") as f:
                print(filename)
                original_data = f["semantic"]
                class_ids = np.hstack([np.unique(original_data), class_ids])
        return np.unique(class_ids)
    
    max_range = np.ones((3, 1)) * (-1e8)
    min_range = np.ones((3, 1)) * 1e8
    record = dict()
    print(find_class())

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


def bin2npy(log_dir: str):
    videos = os.listdir(log_dir + '/sequences')
    videos.sort()
    npy_dir = log_dir + '/predict_npy'
    os.makedirs(npy_dir)
    for video in tqdm(videos):
        os.makedirs(npy_dir + '/' + video)
        frames_root = log_dir + '/sequences/' + video + '/predictions'
        labels = []
        frames = os.listdir(frames_root)
        frames.sort()
        for frame in frames:
            label = np.fromfile(os.path.join(frames_root, frame), dtype=np.int32)
            label = label.reshape((-1))
            labels.append(label)
        labels = np.vstack(labels, dtype=np.int8)
        np.save(npy_dir + '/' + video + '/labels.npy', labels)
    npy_files = os.listdir(npy_dir)
    npy_files.sort()
    data = []
    for npy_file in npy_files:
        npy_data = np.load(os.path.join(npy_dir, npy_file, "labels.npy"))
        data.append(npy_data.reshape(1, 300, 8192))
    data = np.vstack(data, dtype=np.int8)
    np.save(log_dir + '/predict.npy', data)
    print(np.shape(data))


def remap(semantic: np.array, inverse: bool = False, config_path: str = None):
        """
        Remap semantic classes.

        Args:
        semantic: semantic classes to remap.
        inverse: class2num if True, num2class if False. NOTE: See KITTI config for more.
        config_path: path to SemanticKITTI config file.
        """
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
    
    
def visualize_kitti(data_dir: str, log_dir: str, config_path: str, frame_num: int, predict: bool = True):
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
    labels = remap(sem_label, False, config_path)
    
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


def tools(dataset: str, root_dir: str = None):
    if dataset == "hoi4d":
        # config_dir = "../configs/hoi4d"
        # hoi4d_range(root_dir)
        log_dir = "logs_test/log_3Dconv_4Dattention_CPE_RPE_large_test_hoi4d"
        bin2npy(log_dir)
    elif dataset == "kitti":
        config_dir = "../configs/kitti"
        config_path = "../configs/kitti/semantic-kitti-all.yaml"
        log_dir = "../logs/log_MultiScan_3Dconv_4Dattention_test_kitti"
        frame_num = 3600
        visualize_kitti(root_dir, log_dir, config_path, frame_num, predict=True)
        visualize_kitti(root_dir, log_dir, config_path, frame_num, predict=False)
    else:
        assert 0, "dataset not supported!"
    print(f"Dataset {dataset} done.")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=False)
    parser.add_argument("--root_dir", type=str, required=False)
    args = parser.parse_args()
    tools(args.dataset, args.root_dir)


