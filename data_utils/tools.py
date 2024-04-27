

def construct_hoi4d(dataset_dir: str):
    """
    Construct filelist for HOI4D.

    Args:
    dataset_dir: path to save filelist.
    """
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
    import os

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=False, default="../dataset/SemanticKITTI")
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    
    save_dir = f"config/{args.dataset}"
    if args.dataset == "kitti":
        construct_kitti(args.root_dir, save_dir)
    elif args.dataset == "hoi4d":
        construct_hoi4d(save_dir)
