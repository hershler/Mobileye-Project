from data_preparing import prepare_data
import numpy as np
import matplotlib.pyplot as plt
import os

def read_files(data_dir, index):

    data_file = data_dir + 'data.bin'
    label_file = data_dir + 'labels.bin'
    crop_size = (81,81,3)
    data = np.memmap(data_file, dtype='uint8', mode='r', shape=crop_size, offset = crop_size[0]*crop_size[1]*crop_size[2]*index)
    label = np.memmap(label_file, dtype='uint8', mode='r', shape=(1), offset = index)

    plt.imshow(data)
    plt.title("Traffic light" if label else "Not Traffic light")
    
            
def main():



    # read sentences from files
    for (root, dirs, files) in os.walk("../../data/dataset/", topdown=True):
        for file in files:
            print(file)
    root_dir = "../../data/"
    data_dir = root_dir + "dataset/"
    
    #len = prepare_data(root_dir, data_dir, "train/")
    #prepare_data(root_dir, data_dir, "val/")

    #for index in range(1):
    #    read_files(data_dir + "train/", index)
    #    plt.show(block=True)


if __name__ == '__main__':
    main()
