import numpy as np
from operator import itemgetter


def calc_TFL_dist(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    if(abs(tZ) < 10e-6):
        print('tz = ', tZ)
    elif (norm_prev_pts.size == 0):
        print('no prev points')
    elif (norm_prev_pts.size == 0):
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ)
    return curr_container


def prepare_3D_data(prev_container, curr_container, focal, pp):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose(np.array(curr_container.EM))
    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []
    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)
        if not valid:
            Z = 0
        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)
    return corresponding_ind, np.array(pts_3D), validVec


def normalize(pts, focal, pp):
    # transform pixels into normalized pixels using the focal length and principle point
    
    return (pts - pp) / focal

    
def unnormalize(pts, focal, pp):
    # transform normalized pixels into pixels using the focal length and principle point
    
    return pts*focal + pp


def decompose(EM):
    # extract R, foe and tZ from the Ego Motion
    R = EM[:3,:3]
    t = EM[:3,3]
    tX, tY, tZ = t[0], t[1], t[2]
    foe = np.array([tX/tZ, tY/tZ])
    
    return R, foe, tZ


def rotate(pts, R):
    # rotate the points - pts using R
    ones = np.ones((3, 1), int)
    vec_x_y_1 = (np.hstack([pts, ones])).T
    vec_a_b_c = np.dot(R, vec_x_y_1)
    
    return (vec_a_b_c[:2]/vec_a_b_c[2]).T


def find_corresponding_points(p, norm_pts_rot, foe):
    # compute the epipolar line between p and foe
    # run over all norm_pts_rot and find the one closest to the epipolar line
    # return the closest point and its index
    x, y = 0, 1
    m = (foe[y] - p[y])/(foe[x] - p[x])
    n = (p[y]*foe[x] - foe[y]*p[x])/(foe[x] - p[x])
    dist_list = []
    
    for i, pts in enumerate(norm_pts_rot):
        dist = abs((m * pts[x] + n - pts[y]) / np.sqrt(m**2 + 1))
        dist_list.append([dist, i])
    _, idx = min(dist_list, key=itemgetter(0))
    
    return idx, norm_pts_rot[idx]


def calc_dist(p_curr, p_rot, foe, tZ):
    # calculate the distance of p_curr using x_curr, x_rot, foe_x and tZ
    # calculate the distance of p_curr using y_curr, y_rot, foe_y and tZ
    # combine the two estimations and return estimated Z
    x, y = 0, 1
    Z_by_x = tZ*(foe[x] - p_rot[x])/(p_curr[x]-p_rot[x])
    Z_by_y = tZ*(foe[y] - p_rot[y])/(p_curr[y]-p_rot[y])
    
    diff_x = abs(foe[x] - p_curr[x])
    diff_y = abs(foe[y] - p_curr[y])
    ratio = diff_x / (diff_x + diff_y)
    
    return Z_by_x * ratio + Z_by_y * (1 - ratio)