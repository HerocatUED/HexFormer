import os
import h5py
import argparse
import numpy as np
from tqdm import tqdm


def construct_hoi4d(root_dir: str, dataset_dir: str):
    """
    Construct filelist for HOI4D.

    Args:
    root_dir: path to HOI4D seg data.
    dataset_dir: path to save filelist.
    """
    # os.makedirs(a)
    
    train_list = ""
    val_list = ""
    test_list = ""
    
    for filename in ["train1", "train2", "train3", "train4"]:
        with h5py.File(f"{root_dir}/{filename}.h5", "r") as f:
            pass
            
    def data_list(video_id):
        file_list = ""
        for i in range(video_id * 300, video_id * 300 + 300):
            file_list += str(i) + "\n"
        return file_list

    train_list = ""
    val_list = ""
    test_list = ""

    for video_id in range(2971):
        if video_id % 5 > 0:
            train_list += data_list(video_id)
        else:  
            val_list += data_list(video_id)
    
    for video_id in range(500):
        test_list += data_list(video_id)

    f_train = open(f"{dataset_dir}/train_data.txt", "w")
    f_train.write(train_list)
    f_train.close()
    f_val = open(f"{dataset_dir}/val_data.txt", "w")
    f_val.write(val_list)
    f_val.close()
    f_test = open(f"{dataset_dir}/test_data.txt", "w")
    f_test.write(test_list)
    f_test.close()
    
 
def construct_kitti(root_dir: str, dataset_dir: str):
    """
    Construct filelist for KITTI.
    mode: 'train' use 00-10; 'test' use 11-21.

    Args:
    root_dir: path to KITTI.
    dataset_dir: path to save filelist.
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

    f_train = open(f"{dataset_dir}/train_data.txt", "w")
    f_train.write(train_list)
    f_train.close()
    f_val = open(f"{dataset_dir}/val_data.txt", "w")
    f_val.write(val_list)
    f_val.close()
    f_test = open(f"{dataset_dir}/test_data.txt", "w")
    f_test.write(test_list)
    f_test.close()
    

def float64to32(path: str, chunk_size: int = 300):
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
    
    for filename in ["test_float32.h5", "train1_float32.h5", "train2_float32.h5", "train3_float32.h5", "train4_float32.h5"]:
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
                max_range = np.max(np.concatenate([max_range, max_tmp], axis=-1), axis=-1).reshape((3, 1))
                min_range = np.min(np.concatenate([min_range, min_tmp], axis=-1), axis=-1).reshape((3, 1))
                record[filename] = [max_range, min_range]
                
    print(f"max: {max_range}")
    print(f"min: {min_range}")
    print(record)     

if __name__ == "__main__":
    
    hoi4d_range("/mnt/sdc/wangx/HOI4D/HOI4D_dataset/seg_data_h5")
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--root_dir", type=str, required=False, default="../dataset/SemanticKITTI")
    # parser.add_argument("--dataset", type=str, required=True)
    # args = parser.parse_args()
    
    # save_dir = f"config/{args.dataset}"
    # if args.dataset == "kitti":
    #     construct_kitti(args.root_dir, save_dir)
    # elif args.dataset == "hoi4d":
    #     construct_hoi4d(save_dir)
