# Helper functions for data processing, only used for HOI4D
# Helper functions of SemanticKITTI please refer to https://github.com/PRBonn/semantic-kitti-api

import os
import h5py
import argparse
import numpy as np
from tqdm import tqdm


def hoi4d_clip(origin_path: str, clip_path: str, clip_len: int):
    """
    Clip a small chunck of HOI4D.h5 file
    
    Args:
    origin_path: path to original HOI4D.h5 file
    clip_path: path to save cliped file
    clip_len: video number to clip
    """
    with h5py.File(origin_path + "/train1.h5", "r") as f:
        data = f["pcd"][:clip_len]
    with h5py.File(clip_path, "w") as new_f:
        new_f.create_dataset(
            "pcd", shape=data.shape, dtype=np.float32, chunks=True) 
        new_f["pcd"][:] = data.astype(np.float32)


def h5float64to32(origin_path: str, chunk_size: int = 100):
    """
    Convert h5 files from float64 to float32 to save memory.

    Args:
    chunk_size: size of data processed at one time. NOTE set it according to memory of your device.
    origin_path: path to dir containing h5 files of HOI4D.
    """
    print("Converting to float32...")

    for filename in ["train1", "train2", "train3", "train4"]:
        with h5py.File(origin_path + "/" + filename + ".h5", "r") as f:
            with h5py.File(origin_path + "/" + filename + "_float32.h5", "w") as new_f:
                for dataset_name in ["center", "semantic", "pcd"]:

                    original_data = f[dataset_name]
                    print(dataset_name, type(original_data[0].dtype))
                    shape = original_data.shape

                    if dataset_name == "semantic":
                        new_f.create_dataset(
                            dataset_name, shape=shape, dtype=np.int8, chunks=True
                        )
                    else:
                        new_f.create_dataset(
                            dataset_name, shape=shape, dtype=np.float32, chunks=True
                        )

                    total_data = shape[0]
                    num_iterations = total_data // chunk_size

                    for i in tqdm(range(num_iterations)):
                        start_idx = i * chunk_size
                        end_idx = (i + 1) * chunk_size
                        chunk_data = original_data[start_idx:end_idx]
                        if dataset_name == "semantic":
                            single_precision_chunk = chunk_data.astype(np.int8)
                        else:
                            single_precision_chunk = chunk_data.astype(np.float32)
                        new_f[dataset_name][start_idx:end_idx] = single_precision_chunk

                    if total_data % chunk_size != 0:
                        start_idx = num_iterations * chunk_size
                        end_idx = total_data
                        chunk_data = original_data[start_idx:end_idx]
                        single_precision_chunk = chunk_data.astype(np.float32)
                        new_f[dataset_name][start_idx:end_idx] = single_precision_chunk
    print("Done.")
    
    
def hoi4d_range(origin_path: str, chunk_size: int = 20, need_class: bool = False):
    """
    Find data range for hoi4d

    Args:
    chunk_size: size of data processed at one time. NOTE set it according to memory of your device.
    origin_path: path to dir containing h5 files of HOI4D.
    """
    
    def find_range(datas, func):
        rg = np.zeros((len(datas), 1))
        for i, data in enumerate(datas):
            rg[i] = func(data)
        return rg
    
    def find_class():
        class_ids = np.array([-1])
        for filename in ["test.h5", "train1.h5", "train2.h5", "train3.h5", "train4.h5"]:
            with h5py.File(origin_path + "/" + filename, "r") as f:
                print(filename)
                original_data = f["semantic"]
                class_ids = np.hstack([np.unique(original_data), class_ids])
        return np.unique(class_ids)
    
    max_range = np.ones((3, 1)) * (-1e8)
    min_range = np.ones((3, 1)) * 1e8
    record = dict()
    
    if need_class: 
        print(find_class())

    for filename in ["test.h5", "train1.h5", "train2.h5", "train3.h5", "train4.h5"]:
        with h5py.File(origin_path + "/" + filename, "r") as f:
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


def bin2npy(log_dir: str, mode: str):
    """ 
    Convert bin files (structure of KITTI) to npy files (structure of HO4D).
    Only used when we hand in predict results.
    
    Args:
    log_dir: path to inference log dir.
    mode: "sem" means semantic segmentation; "act" means action segmentation.
    """
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
        if mode == "sem":
            data.append(npy_data.reshape(1, 300, 8192))
        elif mode == "act":
            data.append(npy_data.reshape(1, 150))
        else: raise NotImplementedError
    data = np.vstack(data, dtype=np.int8)
    np.save(log_dir + '/predict.npy', data)
    print(np.shape(data))
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--hoi4d_path", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--clip_len", type=int, default=10)
    parser.add_argument("--chunck_size", type=int, default=100)
    args = parser.parse_args()
    
    if args.task == "clip":
        hoi4d_clip(args.hoi4d_path, args.save_path, args.clip_len)
    elif args.task == "h5":
        h5float64to32(args.hoi4d_path, args.chunck_size)
    elif args.task == "range":
        hoi4d_range(args.hoi4d_path)
    elif args.task == "npy_sem":
        bin2npy(args.log_dir, "sem")
    elif args.task == "npy_act":
        bin2npy(args.log_dir, "act")
        