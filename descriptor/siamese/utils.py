import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import keras.backend as K

import os
import cv2
import random
import imgaug as ia
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

aug = iaa.OneOf([
    iaa.Fliplr(0.3),
    iaa.Flipud(0.3),
    iaa.Affine(rotate=(90)),
    iaa.Affine(rotate=(180)),
    iaa.Affine(rotate=(270)),
    iaa.Noop()
])

def create_generator(dataset, num_pairs, batch_size, augmentate=True):    
    i = 0
    input_a, input_b, targets = [], [], []
    
    match_file = pd.read_csv(f'{dataset}/m50_{num_pairs}_{num_pairs}_0.txt', delimiter=' ', 
        names=['patchID1', '3DpointID1', 'unused1', 'patchID2','3DpointID2', 'unused2', 'unused3'])
    
    dissimilarities = match_file[match_file['3DpointID1'] != match_file['3DpointID2']]
    similarities = match_file[match_file['3DpointID1'] == match_file['3DpointID2']]
    assert len(similarities) == len(dissimilarities) == len(match_file) // 2 == num_pairs // 2
        
    while True:      
        if i % num_pairs == 0: # shuffle dataset every epoch
            similarities = similarities.sample(frac=1)
            dissimilarities = dissimilarities.sample(frac=1)
        
        if i % batch_size == 0 and i != 0:
            assert len(input_a) == len(input_b) == batch_size
            
            if augmentate:
                global aug
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
    img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    assert img is not None
    return img

def read_keypoint_patch(img_file, x, y, kp_size):
    img = read_image(img_file)
    x = int(x)
    y = int(y)
    r = max([int(np.ceil(kp_size)) // 2, 32])
    max_y, max_x = img.shape
    r = min([r, np.abs(max_y - y), y, np.abs(max_x - x), x])
    return cv2.resize(img[y - r : y + r, x - r : x + r], (64, 64))

def create_aug_dataset(src_dir, dest_dir, aug):
    error_files = []
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        
    sift = cv2.xfeatures2d.SIFT_create(100)
    aug_det = aug.to_deterministic()
    
    for img_file in tqdm(os.listdir(src_dir)):
        img_file = os.path.join(src_dir, img_file)
        aug_img_file = os.path.join(dest_dir, os.path.basename(img_file))
        if os.path.exists(aug_img_file):
            continue
        try:
            img = read_image(img_file)
        except AssertionError:
            error_files.append(img_file)
            continue
            
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

        kp_file['aug_img_file'] = aug_img_file
        kp_file['src_img_file'] = img_file

        cv2.imwrite(aug_img_file, aug_img)
        pre, ext = os.path.splitext(aug_img_file)
        kp_file.to_csv(f'{pre}.csv')
   
    return error_files

def _create_matches(kp_files, num_matches):
    input_a, input_b, targets = [], [], []
    for kp_file in kp_files:
        line_sample = kp_file.sample(n=num_matches//len(kp_files))
        for i in range(len(line_sample)):
            line = line_sample.iloc[i]
            input_a.append(read_keypoint_patch(line.src_img_file, line.src_x, line.src_y, line.kp_size).reshape(64, 64, 1))
            input_b.append(read_keypoint_patch(line.aug_img_file, line.aug_x, line.aug_y, line.kp_size).reshape(64, 64, 1))
            targets.append(0)
    return input_a, input_b, targets

def _create_nonmatches(kp_files, num_nonmatches):
    input_a, input_b, targets = [], [], []
    num_files = len(kp_files)
    for i in range(num_files):
        kp_file = kp_files[i]
        another_kp_file = kp_files[(i + 2) % num_files]
        line_sample = kp_file.sample(n=num_nonmatches//num_files)
        another_line_sample = another_kp_file.sample(n=num_nonmatches//num_files)
        for j in range(len(line_sample)):
            line = line_sample.iloc[j]
            another_line = another_line_sample.iloc[j]
            input_a.append(read_keypoint_patch(line.src_img_file, line.src_x, line.src_y, line.kp_size).reshape(64, 64, 1))
            input_b.append(read_keypoint_patch(another_line.src_img_file, another_line.src_x, another_line.src_y, another_line.kp_size).reshape(64, 64, 1))
            targets.append(1) 
    return input_a, input_b, targets
        
def create_text_patches_generator(csv_files, batch_size, augmentate=True):
    while True:
        random.shuffle(csv_files)
        random_csv_files = csv_files[:8]
        kp_files = [pd.read_csv(csv_file, index_col=0) for csv_file in random_csv_files]

        input_a, input_b, targets = _create_matches(kp_files, batch_size // 2)
        input_a_nm, input_b_nm, targets_nm = _create_nonmatches(kp_files, batch_size // 2)
        
        input_a.extend(input_a_nm)
        input_b.extend(input_b_nm)
        targets.extend(targets_nm)
        
        if augmentate:
            global aug
            input_a = aug.augment_images(input_a) 
            input_b = aug.augment_images(input_b) 
        
        yield [np.asarray(input_a), np.asarray(input_b)], targets
    