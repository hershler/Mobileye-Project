
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

   
def load_data():

    images_url = r"../../data/leftImg8bit_trainvaltest/leftImg8bit/train/aachen"
    image_list = glob.glob(os.path.join(images_url, '*_leftImg8bit.png'))

    labels_url = r"../../data/gtFine_trainvaltest/gtFine/train/aachen"
    label_list = glob.glob(os.path.join(labels_url, "*_labelIds.png"))
     
    return image_list, label_list


def pass_over_lists(image_list, label_list):

    for i in range(len(image_list)):
        crop_tfl(image_list[i], label_list[i]) 


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


def crop_image(image, x_coords, y_coords, size):

    x_coords += size
    y_coords += size
    index = random.randrange(len(x_coords))
    result_crop = image[x_coords[index] - size:x_coords[index] + size + 1, y_coords[index] - size:y_coords[index] + size + 1]

    return result_crop


def crop_tfl(image_url, label_url): 

    image = np.array(Image.open(image_url))
    label = np.array(Image.open(label_url))

    pixels_of_tfl = np.where(label == 19)
    pixels_not_of_tfl = np.where(label != 19)

    crop_size = 81

    if(len(pixels_of_tfl[0])):
    
        image, label = padding(image, label, crop_size // 2)
        
        data_file = r'../../data/dataset/data.bin'
        labels_file = r'../../data/dataset/labels.bin'
        
        is_tfl = np.ones(1)
        is_not_tfl = np.zeros(1)

        with open(data_file, mode='wb') as data_obj:

            with open(labels_file, mode='wb') as labels_obj:

                result = crop_image(image, pixels_of_tfl[0], pixels_of_tfl[1], crop_size // 2)
                result.tofile(data_obj)
                labels_obj.write(bytes(is_tfl))
                
                result = crop_image(image, pixels_not_of_tfl[0], pixels_not_of_tfl[1], crop_size // 2)
                result.tofile(data_obj)
                labels_obj.write(bytes(is_not_tfl))


def main():

    image_list, label_list =  load_data()
    pass_over_lists(image_list, label_list)
    

if __name__ == '__main__':
    main()
