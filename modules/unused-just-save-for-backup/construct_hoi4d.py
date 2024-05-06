# Convert dataset file to another format

import os
import h5py
import numpy as np
from tqdm import tqdm


def h5float64to32(path: str, chunk_size: int = 100):
    """
    Convert h5 files from float64 to float32 to save memory.

    Args:
    chunk_size: size of data processed at one time. NOTE set it according to memory of your device.
    path: path to dir containing h5 files of HOI4D.
    """
    print("Converting to float32...")

    for filename in ["train1", "train2", "train3", "train4"]:
        with h5py.File(path + "/" + filename + ".h5", "r") as f:
            with h5py.File(path + "/" + filename + "_float32.h5", "w") as new_f:
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


def construct_train(
    h5path: list,
    dataset_name: str,
    clip_length: int,
    n_video: int = -1,
    n_frame: int = -1,
    chunk_size: int = 50,
):
    """
    Construct dataset and config files.
    Args:
    h5path: list of path to .h5 file of hoi4d dataset.
    dataset_name: name of folder to save clipped file, $batch size$ will be number of clipped file to read in a batch.
    clip_length: number of frame in a single clip.
    n_video: number of video to load from single h5 file.
    n_frame: number of frame to load from single video.
    chunk_size: size of data processed at one time. NOTE set it according to memory of your device.
    """
    print(f"Constructing dataset into {dataset_name}...")

    train_list = ""
    val_list = ""
    os.makedirs(dataset_name, exist_ok=True)

    for k, path in enumerate(h5path):

        f = h5py.File(path, "r")
        all_video = f["pcd"].shape[0] if n_video == -1 else n_video
        all_frame = f["pcd"].shape[1] if n_frame == -1 else n_frame
        print(f"Loading {path}...")
        print(f"video num: {all_video}")
        print(f"frames per video: {all_frame}")

        batch = all_video // chunk_size
        delta = 1 if all_video - batch * chunk_size > 0 else 0

        for b in tqdm(range(batch + delta)):
            start_id = b * chunk_size
            if b < batch:
                points_xyz = np.array(
                    f["pcd"][start_id : start_id + chunk_size, :n_frame]
                )
                semantic = np.array(
                    f["semantic"][start_id : start_id + chunk_size, :n_frame]
                )
            else:
                points_xyz = np.array(f["pcd"][start_id:, :n_frame])
                semantic = np.array(f["semantic"][start_id:, :n_frame])

            # Converting xyz to txyz
            video_num, t_video, n_point, _ = points_xyz.shape
            points_txyz = np.concatenate(
                [np.zeros([video_num, t_video, n_point, 1]), points_xyz], axis=-1
            )
            points_txyz[:, :, :, 0] += (np.arange(t_video) + 1)[None, :, None]
            assert len(points_txyz) == len(
                semantic
            ), "semantic should be corresponding to points_txyz"

            # loop over videos
            for v in range(video_num):

                v_ = start_id + v
                t_video, _, _ = points_txyz[v].shape
                N = t_video // clip_length

                pcd = points_txyz[v][: N * clip_length].reshape((N, -1, 4))
                label = semantic[v][: N * clip_length].reshape((N, -1))

                for i in range(N):
                    np.savez(
                        f"{dataset_name}/data_{k+1}_{v_}_{i}.npz",
                        points=pcd[i],
                        labels=label[i],
                    )
                    if v % 5 == 0:
                        val_list += f"{dataset_name}/data_{k+1}_{v_}_{i}.npz\n"
                    else:
                        train_list += f"{dataset_name}/data_{k+1}_{v_}_{i}.npz\n"

                # left frames
                if t_video % clip_length > 0:
                    pcd = points_txyz[v][N * clip_length :].reshape((-1, 4))
                    label = semantic[v][N * clip_length :].reshape((-1))
                    np.savez(
                        f"{dataset_name}/data_{k+1}_{v_}_{N}.npz",
                        points=pcd,
                        labels=label,
                    )
                    train_list += f"{dataset_name}/data_{k+1}_{v_}_{N}.npz\n"

    # save config file
    f_train = open(f"{dataset_name}/train_data.txt", "w")
    f_val = open(f"{dataset_name}/val_data.txt", "w")
    f_train.write(train_list)
    f_train.close()
    f_val.write(val_list)
    f_val.close()


def construct_test(
    h5path: list, dataset_name: str, clip_length: int, chunk_size: int = 50
):
    """
    Construct dataset and config files.
    Args:
    h5path: list of path to .h5 file of hoi4d dataset.
    dataset_name: name of folder to save clipped file, $batch size$ will be number of clipped file to read in a batch.
    clip_length: number of frame in a single clip.
    chunk_size: size of data processed at one time. NOTE set it according to memory of your device.
    """
    print(f"Constructing dataset into {dataset_name}...")

    test_list = ""
    os.makedirs(dataset_name, exist_ok=True)

    for k, path in enumerate(h5path):

        f = h5py.File(path, "r")
        all_video = f["pcd"].shape[0]
        all_frame = f["pcd"].shape[1]
        print(f"Loading {path}...")
        print(f"video num: {all_video}")
        print(f"frames per video: {all_frame}")

        batch = all_video // chunk_size
        delta = 1 if all_video - batch * chunk_size > 0 else 0

        for b in tqdm(range(batch + delta)):
            start_id = b * chunk_size
            if b < batch:
                points_xyz = np.array(f["pcd"][start_id : start_id + chunk_size])
            else:
                points_xyz = np.array(f["pcd"][start_id:])

            # Converting xyz to txyz
            video_num, t_video, n_point, _ = points_xyz.shape
            points_txyz = np.concatenate(
                [np.zeros([video_num, t_video, n_point, 1]), points_xyz], axis=-1
            )
            points_txyz[:, :, :, 0] += (np.arange(t_video) + 1)[None, :, None]

            # loop over videos
            for v in range(video_num):

                v_ = start_id + v
                t_video, _, _ = points_txyz[v].shape
                N = t_video // clip_length

                pcd = points_txyz[v][: N * clip_length].reshape((N, -1, 4))

                for i in range(N):
                    np.savez(f"{dataset_name}/data_{k+1}_{v_}_{i}.npz", points=pcd[i])
                    if v % 5 == 0:
                        val_list += f"{dataset_name}/data_{k+1}_{v_}_{i}.npz\n"
                    else:
                        test_list += f"{dataset_name}/data_{k+1}_{v_}_{i}.npz\n"

                # left frames
                if t_video % clip_length > 0:
                    pcd = points_txyz[v][N * clip_length :].reshape((-1, 4))
                    np.savez(f"{dataset_name}/data_{k+1}_{v_}_{N}.npz", points=pcd)
                    test_list += f"{dataset_name}/data_{k+1}_{v_}_{N}.npz\n"

    # save config file
    f_test = open(f"{dataset_name}/test_data.txt", "w")
    f_test.write(test_list)
    f_test.close()


if __name__ == "__main__":

    # example of building dataset using HOI4D
    clip_length = 8
    train_path = [
        "/mnt/sdc/wangx/HOI4D/HOI4D_dataset/seg_data_h5/train1_float32.h5",
        "/mnt/sdc/wangx/HOI4D/HOI4D_dataset/seg_data_h5/train2_float32.h5",
        "/mnt/sdc/wangx/HOI4D/HOI4D_dataset/seg_data_h5/train3_float32.h5",
        "/mnt/sdc/wangx/HOI4D/HOI4D_dataset/seg_data_h5/train4_float32.h5",
    ]
    test_path = ["/mnt/sdc/wangx/HOI4D/HOI4D_dataset/seg_data_h5/test_float32.h5"]
    trainset_name = f"/mnt/sdc/wangx/HexFormer/dataset/hoi4d/frame{clip_length}_full"
    testset_name = f"/mnt/sdc/wangx/HexFormer/dataset/hoi4d/frame{clip_length}_test"
    # h5float64to32(h5_dir, 100) NOTE run only if you need to save memory
    # construct_train(train_path, trainset_name, clip_length)
    construct_test(test_path, testset_name, clip_length)
