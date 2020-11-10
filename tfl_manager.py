import pickle

from attention_phase1.attention import find_tfl_lights
from sfm_phase3.SFM import calc_tfl_dist, normalize, unnormalize, rotate, prepare_3D_data
from detection_phase2.data_preparing import crop_image
from frame_container import FrameContainer

from tensorflow.keras.models import load_model

import numpy as np
import matplotlib.pyplot as plt
# import random

DEBUG = True
# TODO: add asserts to check validity


class TFLMan:

    def __init__(self, pkl_path):

        with open(pkl_path, 'rb') as pkl_file:
            self.__data = pickle.load(pkl_file, encoding='latin1')
        self.__pp = self.__data['principle_point']
        self.__focal = self.__data['flx']
        self.__prev_container = None
        self.curr_container = None
        self.__net = load_model(r"data/model.h5")

        if DEBUG is True:
            self.tfl_candidates, self.tfl, self.tfl_distance = None, None, None

    def run(self, curr_image, _id):

        if DEBUG is True:
            fig, (self.tfl_candidates, self.tfl, self.tfl_distance) =\
                plt.subplots(1, 3, figsize=(12, 5))

        self.curr_container = FrameContainer(curr_image)

        candidates, auxiliary = self.__get_tfl_candidates()
        self.curr_container.traffic_light, tfl_aux = self.__get_tfl_coordinates(candidates, auxiliary)
        self.__get_distance(_id)

        if DEBUG is True:
            plt.show()

        self.__prev_container = self.curr_container

        return self.curr_container.traffic_lights_3d_location, tfl_aux

    def __get_tfl_candidates(self):
        x_red, y_red, x_green, y_green = find_tfl_lights(self.curr_container.img)

        candidates = [[x_red[i], y_red[i]] for i in range(len(x_red))] + \
                     [[x_green[i], y_green[i]] for i in range(len(x_green))]
        auxiliary = [0] * len(x_red) + [1] * len(x_green)

        if DEBUG is True:
            self.visualize1(x_green, x_red, y_green, y_red)

        return np.array(candidates), auxiliary

    def __get_tfl_coordinates(self, candidates, auxiliary):
        crop_size = 81
        l_predicted_label = []

        for candidate in candidates:
            crop_img = crop_image(self.curr_container.img, candidate, crop_size, padded=False)
            predictions = self.__net.predict(crop_img.reshape([-1, crop_size, crop_size, 3]))
            # predictions = [[0, random.random() + 0.2]]
            l_predicted_label.append(1 if predictions[0][1] > 0.98 else 0)

        traffic_lights = [candidates[i] for i in range(len(candidates)) if l_predicted_label[i] == 1]
        auxiliary = [auxiliary[i] for i in range(len(auxiliary)) if l_predicted_label[i] == 1]

        if DEBUG is True:
            self.visualize2(auxiliary, traffic_lights)

        return np.array(traffic_lights), auxiliary

    def __get_distance(self, _id):

        if len(self.curr_container.traffic_light) and self.__prev_container:
            EM = np.eye(4)
            for i in range(_id - 1, _id):
                EM = np.dot(self.__data['egomotion_' + str(i) + '-' + str(i + 1)], EM)
            self.curr_container.EM = EM
            tfl_3D_location = calc_tfl_dist(self.__prev_container, self.curr_container, self.__focal, self.__pp)
            self.curr_container.traffic_light_3D_location = tfl_3D_location

        if DEBUG is True:
            self.visualize3()

    def visualize1(self, x_green, x_red, y_green, y_red):
        self.tfl_candidates.set_title('tfl candidates:')
        self.tfl_candidates.imshow(self.curr_container.img)
        self.tfl_candidates.plot(x_red, y_red, 'r+')
        self.tfl_candidates.plot(x_green, y_green, 'g+')

    def visualize2(self, auxiliary, traffic_lights):
        self.tfl.set_title('tfl 2D location:')
        self.tfl.imshow(self.curr_container.img)
        for aux, coord in zip(auxiliary, traffic_lights):
            self.tfl.plot(coord[0], coord[1], 'g+' if 1 == aux else 'r+')

    def visualize3(self):

        self.tfl_distance.set_title('tfl 3D location:')
        self.tfl_distance.imshow(self.curr_container.img)
        if not len(self.curr_container.traffic_light) or not self.__prev_container:
            return
        tfl_3d_coordinates = zip(self.curr_container.traffic_light[:, 0],
                                 self.curr_container.traffic_light[:, 1],
                                 self.curr_container.traffic_lights_3d_location[:, 2])
        norm_prev_pts, norm_curr_pts, R, norm_foe, tZ = prepare_3D_data(self.__prev_container, self.curr_container,
                                                                        self.__focal, self.__pp)
        norm_rot_pts = rotate(norm_prev_pts, R)
        rot_pts = unnormalize(norm_rot_pts, self.__focal, self.__pp)
        foe = np.squeeze(unnormalize(np.array(norm_foe), self.__focal, self.__pp))
        self.tfl_distance.plot(rot_pts[:, 0], rot_pts[:, 1], 'g+')
        self.tfl_distance.plot(foe[0], foe[1], 'r+', markersize=20)
        for i, (x, y, z) in enumerate(tfl_3d_coordinates):
            self.tfl_distance.plot([x, foe[0]], [y, foe[1]], 'b')

            if self.curr_container.valid[i]:
                self.tfl_distance.text(x, y, r'{0:.1f}'.format(z), color='b', fontsize=12)
        self.tfl_distance.plot(self.curr_container.traffic_light[:, 0],
                               self.curr_container.traffic_light[:, 1], 'g+')
