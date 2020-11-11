import numpy as np
import pickle
import matplotlib.pyplot as plt
import SFM
from frame_container import FrameContainer

curr_frame_id = 25
prev_frame_id = 24


def visualize(prev_container: FrameContainer, curr_container: FrameContainer, focal: np.ndarray, pp: np.ndarray)\
        -> None:
    norm_prev_pts, norm_curr_pts, R, norm_foe, tZ = SFM.prepare_3D_data(prev_container, curr_container, focal, pp)
    norm_rot_pts = SFM.rotate(norm_prev_pts, R)
    rot_pts = SFM.unnormalize(norm_rot_pts, focal, pp)
    foe = np.squeeze(SFM.unnormalize(np.array([norm_foe]), focal, pp))

    fig, (curr_sec, prev_sec) = plt.subplots(1, 2, figsize=(12, 6))
    prev_sec.set_title('prev(' + str(prev_frame_id) + ')')
    prev_sec.imshow(prev_container.img)
    prev_p = prev_container.traffic_light
    prev_sec.plot(prev_p[:, 0], prev_p[:, 1], 'b+')

    curr_sec.set_title('curr(' + str(curr_frame_id) + ')')
    curr_sec.imshow(curr_container.img)
    curr_p = curr_container.traffic_light
    curr_sec.plot(curr_p[:, 0], curr_p[:, 1], 'b+')

    for i in range(len(curr_p)):
        curr_sec.plot([curr_p[i, 0], foe[0]], [curr_p[i, 1], foe[1]], 'b')
        if curr_container.valid[i]:
            curr_sec.text(curr_p[i, 0], curr_p[i, 1],
                          r'{0:.1f}'.format(curr_container.traffic_lights_3d_location[i, 2]), color='r')
    curr_sec.plot(foe[0], foe[1], 'r+')
    curr_sec.plot(rot_pts[:, 0], rot_pts[:, 1], 'g+')
    plt.show()


def test_calc_tfl_dist() -> None:
    pkl_path = 'dusseldorf_000049.pkl'
    prev_img_path = 'dusseldorf_000049_0000' + str(prev_frame_id) + '_leftImg8bit.png'
    curr_img_path = 'dusseldorf_000049_0000' + str(curr_frame_id) + '_leftImg8bit.png'
    prev_container = FrameContainer(prev_img_path)
    curr_container = FrameContainer(curr_img_path)
    with open(pkl_path, 'rb') as pkl_file:
        data = pickle.load(pkl_file, encoding='latin1')
    focal = data['flx']
    pp = data['principle_point']
    prev_container.traffic_light = np.array(data['points_' + str(prev_frame_id)][0])
    curr_container.traffic_light = np.array(data['points_' + str(curr_frame_id)][0])
    curr_container.EM = SFM.calc_EM(data, prev_frame_id, curr_frame_id)
    curr_container = SFM.calc_tfl_dist(prev_container, curr_container, focal, pp)
    visualize(prev_container, curr_container, focal, pp)


if __name__ == '__main__':
    test_calc_tfl_dist()
