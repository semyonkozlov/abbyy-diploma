import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from functools import lru_cache

@lru_cache(maxsize=None)
def read_bitmap(dataset_name, bitmap_index):
    return plt.imread(f'{dataset_name}/patches{bitmap_index:04d}.bmp')
    
def read_patch(dataset_name, patch_index):
    #TODO check if patches are enumerated from 0 or from 1
    return cut_patch_from_bitmap(read_bitmap(dataset_name, patch_index // (16 * 16)), patch_index % (16 * 16))
    
def cut_patch_from_bitmap(bitmap, image_index):
    image_col = image_index % 16
    image_row = image_index // 16

    return bitmap[image_row * 64 : (image_row + 1) * 64, image_col * 64 : (image_col + 1) * 64]

def get_train_generator(dataset, num_pairs, batch_size=128):    
    i = 0
    input_a, input_b, targets = [], [], []
    
    match_file = pd.read_csv(f'{dataset}/m50_{num_pairs}_{num_pairs}_0.txt', delimiter=' ', 
        names=['patchID1', '3DpointID1', 'unused1', 'patchID2','3DpointID2', 'unused2', 'unused3'])
    
    while True:
        if i % batch_size == 0 and i != 0:
            yield [np.array(x).reshape(batch_size, 64, 64, 1) for x in [input_a, input_b]], targets
            input_a, input_b, targets = [], [], []
            
        line = match_file.iloc[i % num_pairs]
        input_a.append(read_patch(dataset, line['patchID1']))
        input_b.append(read_patch(dataset, line['patchID2']))
        targets.append(1 if line['3DpointID1'] == line['3DpointID2'] else 0)
        
        i += 1
        
def get_test_generator(dataset, num_pairs, batch_size=128):
    i = 0
    input_a, input_b = [], []
    
    match_file = pd.read_csv(f'{dataset}/m50_{num_pairs}_{num_pairs}_0.txt', delimiter=' ', 
        names=['patchID1', '3DpointID1', 'unused1', 'patchID2','3DpointID2', 'unused2', 'unused3'])
    
    while True:
        if i % batch_size == 0 and i != 0:
            yield [np.array(x).reshape(batch_size, 64, 64, 1) for x in [input_a, input_b]], targets
            input_a, input_b = [], []
            
        line = match_file.iloc[i % num_pairs]
        input_a.append(read_patch(dataset, line['patchID1']))
        input_b.append(read_patch(dataset, line['patchID2']))
        
        i += 1
    