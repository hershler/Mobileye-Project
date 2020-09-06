
"""Elementary imports: """
import os
import json
import glob
import argparse
import random

"""numpy/scipy imports:"""
import numpy as np
from scipy import signal as sg
import scipy.ndimage as ndimage


"""PIL imports:"""
from PIL import Image

"""matplotlib imports:"""
import matplotlib.pyplot as plt

#from .attention_phase1.run_attention import find_tfl_lights


def padding(image, label, padding_size):

    height, width = label.shape

    v_padding = np.zeros((padding_size, width), int)
    label = np.vstack([v_padding, label, v_padding])
    h_padding = np.zeros((height + padding_size*2, padding_size), int)
    label = np.hstack([h_padding, label, h_padding])
    
    v_padding = np.zeros((padding_size, width, 3), int)
    image = np.vstack([v_padding, image, v_padding])
    h_padding = np.zeros((height + padding_size*2, padding_size, 3), int)
    image = np.hstack([h_padding, image, h_padding])
    
    return image, label


def select_rand_crop_image(image, x_coords, y_coords, size):

    x_coords += size
    y_coords += size
    index = random.randrange(len(x_coords))
    result_crop = image[x_coords[index] - (size + 1):x_coords[index] + size, y_coords[index] - (size + 1):y_coords[index] + size]

    return result_crop


def save_files(data_file, label_file, crop_image, label):

    with open(data_file, mode='ab') as data_obj:
        np.array(crop_image, dtype=np.uint8).tofile(data_obj)

    with open(label_file, mode='ab') as label_obj:
        label_obj.write(bytes(label, 'utf-8'))


def crops_tfl_not_tfl(image_url, label_url):

    image = np.array(Image.open(image_url))
    label = np.array(Image.open(label_url))

    pixels_of_tfl = np.where(label == 19)
    pixels_not_of_tfl = np.where(label != 19)

    crop_size = 81

    if(len(pixels_of_tfl[0])):
    
        image, label = padding(image, label, crop_size // 2)
                
        crop_tfl = select_rand_crop_image(image, pixels_of_tfl[0], pixels_of_tfl[1], crop_size // 2)
        crop_not_tfl = select_rand_crop_image(image, pixels_not_of_tfl[0], pixels_not_of_tfl[1], crop_size // 2)

        return crop_tfl, crop_not_tfl
      
    return None


def load_data(image_dir, label_dir, type_):

    im_cyties_dirs = glob.glob(os.path.join(image_dir + type_, '*'))
    lab_cyties_dirs = glob.glob(os.path.join(label_dir + type_, '*'))
    image_list = []
    label_list = []
    
    for city in im_cyties_dirs:
        image_list += glob.glob(os.path.join(city, '*_leftImg8bit.png'))

    for city in lab_cyties_dirs:
        label_list += glob.glob(os.path.join(city, "*_labelIds.png"))

    return image_list, label_list


def prepare_data(root_dir, data_dir, type_):

    image_list, label_list =  load_data(root_dir + "leftImg8bit_trainvaltest/leftImg8bit/", root_dir + "gtFine_trainvaltest/gtFine/", type_)
    data_dir += type_
    data_file = data_dir + 'data.bin'
    label_file = data_dir + 'labels.bin'

    if os.path.exists(data_file):
        os.remove(data_file)

    if os.path.exists(label_file):
        os.remove(label_file)

    for i in range(len(image_list)):
        crops = crops_tfl_not_tfl(image_list[i], label_list[i])
 
        if(crops):
            is_tfl, is_not_tfl = "1", "0"
            crop_tfl, crop_not_tfl = crops[0], crops[1]

            save_files(data_file, label_file, crop_tfl, is_tfl)
            save_files(data_file, label_file, crop_not_tfl, is_not_tfl)
            
    return len(image_list) + len(label_list)
