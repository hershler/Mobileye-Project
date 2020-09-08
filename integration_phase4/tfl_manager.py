import pickle
from frame_container import FrameContainer
import numpy as np
from run_attention import find_tfl_lights
from SFM import calc_TFL_dist


class TFL_Man:
    def __init__(self, pkl_path):
        with open(pkl_path, 'rb') as pklfile:
            self.data = pickle.load(pklfile, encoding='latin1')

        self.pp = self.data['principle_point']
        self.focal = self.data['flx']
        self.prev_container = FrameContainer()
        self.net = []

    def run(self, curr_frame, _id):

        curr_container = FrameContainer(curr_frame)

        if self.prev_container:
            curr_container.EM = np.dot(self.data['egomotion_' + str(_id) + '-' + str(_id + 1)], np.eye(4))
            curr_container.traffic_light = find_tfl_lights(curr_container.img)
            curr_container.traffic_light = ???
            curr_container = calc_TFL_dist(self.prev_container, curr_container, self.focal, self.pp)

        self.prev_container = curr_container
