import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import keras.backend as K
from functools import lru_cache
from imgaug import augmenters as iaa

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

def create_generator(dataset, num_pairs, batch_size, augmentate=True):    
    i = 0
    input_a, input_b, targets = [], [], []
    
    match_file = pd.read_csv(f'{dataset}/m50_{num_pairs}_{num_pairs}_0.txt', delimiter=' ', 
        names=['patchID1', '3DpointID1', 'unused1', 'patchID2','3DpointID2', 'unused2', 'unused3'])
    
    dissimilarities = match_file[match_file['3DpointID1'] != match_file['3DpointID2']]
    similarities = match_file[match_file['3DpointID1'] == match_file['3DpointID2']]
    assert len(similarities) == len(dissimilarities) == len(match_file) // 2 == num_pairs // 2
    
    aug = iaa.OneOf([
        iaa.Fliplr(0.3),
        iaa.Flipud(0.3),
        iaa.Affine(rotate=(90)),
        iaa.Affine(rotate=(180)),
        iaa.Affine(rotate=(270)),
        iaa.Noop()
    ])
        
    while True:      
        if i % num_pairs == 0: # shuffle dataset every epoch
            similarities = similarities.sample(frac=1)
            dissimilarities = dissimilarities.sample(frac=1)
        
        if i % batch_size == 0 and i != 0:
            assert len(input_a) == len(input_b) == batch_size
            
            if augmentate:
                input_a = aug.augment_images(input_a) 
                input_b = aug.augment_images(input_b) 
                
            yield [np.asarray(input_a), np.asarray(input_b)], targets
            input_a, input_b, targets = [], [], []
            
        line = (similarities if i % 2 == 0 else dissimilarities).iloc[(i // 2) % (num_pairs // 2)]
        input_a.append(read_patch(dataset, line['patchID1']).reshape(64, 64, 1))
        input_b.append(read_patch(dataset, line['patchID2']).reshape(64, 64, 1))
        targets.append(0 if line['3DpointID1'] == line['3DpointID2'] else 1)
        
        i += 1
        
def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))     