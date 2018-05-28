import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import keras.backend as K

import os
import cv2
from imgaug import augmenters as iaa
from tqdm import tqdm

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
        
def show_images(images, n_images_in_row=4, titles=None):
    assert (titles is None) or (len(images) == len(titles))
    n_images = len(images)
    if titles is None:
        titles = [f'Patch {i}' for i in range(1, n_images + 1)]
    n_cols = n_images_in_row
    n_rows = n_images // 4 + 1
    fig = plt.figure(figsize=(n_cols * 4, n_rows * 4))
    for i, (image, title) in enumerate(zip(images, titles)):
        splt = fig.add_subplot(n_rows, n_cols, i + 1)
        plt.imshow(image.squeeze(), cmap='gray')
        splt.set_title(title)
    plt.show()
    
@lru_cache(maxsize=8)
def read_image(img_file):
    return cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_RGB2GRAY)

def read_keypoint_patch(img_file, x, y, kp_size):
    img = read_image(img_file)
    x = int(x)
    y = int(y)
    r = max([int(np.ceil(kp_size)) // 2, 32])
    return cv2.resize(img[y - r : y + r, x - r : x + r], (64, 64))

def create_aug_dataset(src_dir, dest_dir, aug):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        
    sift = cv2.xfeatures2d.SIFT_create(100)
    aug_det = aug.to_deterministic()
    
    for img_file in tqdm(os.listdir(src_dir)):
        img = read_image(img_file)
        aug_img = aug.augment_images([img])[0]

        sift_kps = sift.detect(img)
        kps = ia.KeypointsOnImage([ia.Keypoint(x=int(kp.pt[0]), y=int(kp.pt[1])) for kp in sift_kps], shape=img.shape)
        aug_kps = aug.augment_keypoints([kps])[0]

        kp_file = pd.DataFrame(columns=['src_img_file', 'src_x', 'src_y', 'aug_img_file', 'aug_x', 'aug_y', 'kp_size'])

        src_xy = kps.get_coords_array()
        aug_xy = aug_kps.get_coords_array()
        kp_file['src_x'] = src_xy[:, 0]
        kp_file['src_y'] = src_xy[:, 1]
        kp_file['aug_x'] = aug_xy[:, 0]
        kp_file['aug_y'] = aug_xy[:, 1]
        kp_file['kp_size'] = [int(np.ceil(kp.size)) for kp in sift_kps]

        aug_img_file = os.path.join(dest_dir, os.path.basename(img_file))
        kp_file['aug_img_file'] = aug_img_file
        kp_file['src_img_file'] = img_file

        cv2.imwrite(aug_img_file, aug_img)
        pre, ext = os.path.splitext(aug_img_file)
        kp_file.to_csv(f'{pre}.csv')
    