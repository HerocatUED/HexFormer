import h5py
import numpy as np
from tqdm import tqdm


chunk_size = 10

for filename in ['train1', 'train2', 'train3', 'train4']:
    with h5py.File('../HOI4D_dataset/seg_data_h5'+'/'+filename+'.h5', 'r') as f:
        with h5py.File('../HOI4D_dataset/seg_data_h5'+'/'+filename+'_float32.h5', 'w') as new_f:
            for dataset_name in f.keys():
                print(dataset_name)

                original_data = f[dataset_name]
                print(type(original_data[0].dtype))

                shape = original_data.shape

                new_f.create_dataset(dataset_name, shape=shape, dtype=np.float32, chunks=True)

                total_data = shape[0]
                num_iterations = total_data // chunk_size

                for i in tqdm(range(num_iterations)):
                    start_idx = i * chunk_size
                    end_idx = (i + 1) * chunk_size

                    chunk_data = original_data[start_idx:end_idx]
                    single_precision_chunk = None
                    if dataset_name == 'semantic':
                        single_precision_chunk = chunk_data.astype(np.int16)
                    else: single_precision_chunk = chunk_data.astype(np.float32)

                    new_f[dataset_name][start_idx:end_idx] = single_precision_chunk

                if total_data % chunk_size != 0:
                    start_idx = num_iterations * chunk_size
                    end_idx = total_data

                    chunk_data = original_data[start_idx:end_idx]
                    single_precision_chunk = chunk_data.astype(np.float32)

                    new_f[dataset_name][start_idx:end_idx] = single_precision_chunk

print("Done.")