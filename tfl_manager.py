import pickle

from attention_phase1.attention import find_tfl_lights
from sfm_phase3.SFM import calc_tfl_dist, normalize, unnormalize, rotate, prepare_3D_data
from detection_phase2.data_preparing import crop_image
from frame_container import FrameContainer

import numpy as np
import random

# TODO: add asserts to check validity


class TFLMan:

    def __init__(self, pkl_path):

        with open(pkl_path, 'rb') as pkl_file:
            self.__data = pickle.load(pkl_file, encoding='latin1')
        self.__pp = self.__data['principle_point']
        self.__focal = self.__data['flx']
        self.__prev_container = None
        self.curr_container = None

    def run(self, curr_image, _id):

        self.curr_container = FrameContainer(curr_image)

        candidates, auxiliary = self.__get_tfl_candidates()
        self.curr_container.traffic_light, tfl_aux = self.__get_tfl_coordinates(candidates, auxiliary)
        self.__get_distance(_id)

        self.__prev_container = self.curr_container

        return self.curr_container.traffic_lights_3d_location, tfl_aux

    def __get_tfl_candidates(self):
        x_red, y_red, x_green, y_green = find_tfl_lights(self.curr_container.img)

        candidates = [[x_red[i], y_red[i]] for i in range(len(x_red))] + \
                     [[x_green[i], y_green[i]] for i in range(len(x_green))]
        auxiliary = [0] * len(x_red) + [1] * len(x_green)

        return np.array(candidates), auxiliary

    def __get_tfl_coordinates(self, candidates, auxiliary):
        crop_size = 81
        l_predicted_label = []

        for candidate in candidates:
            crop_img = crop_image(self.curr_container.img, candidate, crop_size, padded=False)
            predictions = [[0, random.random() + 0.2]]
            l_predicted_label.append(1 if predictions[0][1] > 0.98 else 0)

        traffic_lights = [candidates[i] for i in range(len(candidates)) if l_predicted_label[i] == 1]
        auxiliary = [auxiliary[i] for i in range(len(auxiliary)) if l_predicted_label[i] == 1]

        return np.array(traffic_lights), auxiliary

    def __get_distance(self, _id):

        if len(self.curr_container.traffic_light) and self.__prev_container:
            EM = np.eye(4)
            for i in range(_id - 1, _id):
                EM = np.dot(self.__data['egomotion_' + str(i) + '-' + str(i + 1)], EM)
            self.curr_container.EM = EM
            tfl_3D_location = calc_tfl_dist(self.__prev_container, self.curr_container, self.__focal, self.__pp)
            self.curr_container.traffic_light_3D_location = tfl_3D_location
