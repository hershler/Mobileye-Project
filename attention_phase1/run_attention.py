"""Elementary imports: """
import os
import json
import glob
import argparse

"""numpy/scipy imports:"""
import numpy as np
from scipy import signal as sg
import scipy.ndimage as ndimage
from skimage.feature import peak_local_max


"""PIL imports:"""
from PIL import Image

"""matplotlib imports:"""
import matplotlib.pyplot as plt


def non_max_suppression(image, size_):
    coord_x, coord_y = [], []

    image_height, image_width = image.shape[:2]

    for i in range(0, image_height - size_, size_):

        for j in range(0, image_width - size_, size_):
        
            max_coord = np.argmax(image[i:i + size_, j:j + size_])
            x_max = max_coord // size_ + i
            y_max = max_coord % size_ + j
            local_max = image[x_max, y_max]

            if local_max > 120:
                coord_x.append(x_max)
                coord_y.append(y_max)

    return coord_x, coord_y
def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    kernel_r = np.array([[-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5],[-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [1, 1, -0.5], [1, 2, 1], [1, 2, 1], [1, 1, 1], [1, 1, 1]])
    kernel_g = np.array([[1, 1, 1], [1, 1, 1], [1, 2, 1],  [1, 2, 1], [1, 1, -0.5],  [1, 1, -0.5], [-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5],[-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5], [-0.5, -0.5, -0.5],])

    im_red = c_image[:, :, 0]
    im_green = c_image[:, :, 1]

    grad_r = sg.convolve2d(im_red, kernel_r, mode='same')
    grad_g = sg.convolve2d(im_green, kernel_g, mode='same')
    
    coords_red = peak_local_max(grad_r, min_distance=20, num_peaks=5)
    coords_green = peak_local_max(grad_g, min_distance=20, num_peaks=5)
    
 #   x_red, y_red = non_max_suppression(max_im, 20)
 #   x_green, y_green = non_max_suppression(max_im, 20)
    
    
    x_red, y_red = coords_red[:, -1], coords_red[:, 0]
    x_green, y_green = coords_green[:, -1], coords_green[:, 0]
    
    return x_red, y_red, x_green, y_green


def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    """
    Run the attention code
    """
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    show_image_and_gt(image, objects, fig_num)

    red_x, red_y, green_x, green_y = find_tfl_lights(image, some_threshold=42)
    plt.plot(red_x, red_y, 'ro', color='r', markersize=2)
    plt.plot(green_x, green_y, 'ro', color='g', markersize=2)


def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""
    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = '../../data/leftImg8bit_trainvaltest/leftImg8bit/train/bremen'

    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))[18:22]
    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')
        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)
    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


if __name__ == '__main__':
    main()
