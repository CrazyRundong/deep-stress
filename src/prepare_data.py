# -*- coding: utf-8 -*-
import mxnet as mx
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage.interpolation import rotate
from datetime import datetime
import os
import re

comsol_dir = '../Data/comsol_source'
data_path = '../Data/consol_crops.npz'
vst_num = 12  # num of vst per axis
crop_size = 224  # follow standard CNN input size
radius_scale_factor = 0.2  # R = grid_size * radius_scale_factor
n_points = int(crop_size * (vst_num + 1) / radius_scale_factor)  # num of points interpolated per axis
grid_x, grid_y = np.mgrid[0:1:complex(0, n_points), 0:1:complex(0, n_points)]
circular_div = 16  # K in paper
feat_num = 500  # num of feature extracted per sample
plot_num = 20  # num of test samples to plot


def load_and_crop(data_set_dir=comsol_dir):
    assert os.path.exists(data_set_dir)
    vst_locate = np.linspace(0., 1., vst_num + 2)[1: -1]
    radius = 1. / (vst_num + 1) / 2. * radius_scale_factor
    tem_list = []  # the X
    stress_list = []  # the y
    for fname in os.listdir(data_set_dir):
        fname = os.path.splitext(fname)[0]
        tem_tokens = fname.split('_')
        if tem_tokens[0] == 'stress':
            continue
        tem_path = os.path.join(data_set_dir, fname + '.txt')
        stress_name = '_'.join(['stress'] + tem_tokens[1:]) + '.txt'
        stress_path = os.path.join(data_set_dir, stress_name)
        assert os.path.exists(stress_path)
        tem_data = np.loadtxt(tem_path, usecols=(0, 1, 3), dtype=np.float32)
        stress_data = np.loadtxt(stress_path, usecols=(0, 1, 3), dtype=np.float32)
        # scan to [0, 1]
        tem_data[:, :2] *= 100
        stress_data[:, :2] *= 100
        for x in vst_locate:
            for y in vst_locate:
                # TODO(Rundong): not Pythonic, I gona crazy by this, believe me...
                current_tem = tem_data[
                    np.logical_and(np.logical_and((x - radius) <= tem_data[:, 0], tem_data[:, 0] <= (x + radius)),
                                   np.logical_and((y - radius) <= tem_data[:, 1], tem_data[:, 1] <= (y + radius)))]
                current_tem[:, :2] -= (x, y)
                current_tem[:, :2] /= 2 * radius
                # rescan tem distribution coordinate
                # roi_bool_idx = np.abs(current_tem[:, :2]) <= 0.2
                # extended_roi_idx = np.hstack((roi_bool_idx,
                #                              np.zeros((current_tem.shape[0], 1), dtype=bool)))
                # extended_not_roi_idx = np.hstack((np.logical_not(roi_bool_idx),
                #                                  np.zeros((current_tem.shape[0], 1), dtype=bool)))
                # current_tem[extended_roi_idx] *= 2.
                # current_tem[extended_not_roi_idx] *= 1. / 3.
                # current_tem[extended_not_roi_idx] += .4 - 1. / 6.

                current_stress = stress_data[
                    np.logical_and(np.logical_and((x - radius) <= stress_data[:, 0], stress_data[:, 0] <= (x + radius)),
                                   np.logical_and((y - radius) <= stress_data[:, 1], stress_data[:, 1] <= (y + radius)))]
                stress_max = current_stress.max()
                tem_list.append(current_tem)
                stress_list.append(stress_max)

    return tem_list, stress_list


def local_interp(tem_list, stress_list, npz_dir=data_path):
    interp_size = 28  # follow MNIST
    xx, yy = np.mgrid[0:1:complex(0, interp_size), 0:1:complex(0, interp_size)]
    interped_list = []
    for tem in tem_list:
        tem[:, :2] += 0.5  # norm to [0, 1]
        c = griddata(tem[:, :2], tem[:, -1], (xx, yy), method='cubic', fill_value=tem[:, -1].mean())
        interped_list.append(c)
    interped_list = np.array(interped_list)
    stress_list = np.array(stress_list)

    tem_mean = np.mean(interped_list, axis=0)
    interped_list -= tem_mean
    tem_max = np.abs(interped_list).max()
    interped_list /= tem_max

    stress_max = stress_list.max()
    stress_list /= stress_max

    # Dump npz
    np.savez_compressed(npz_dir,
                        temp=interped_list,
                        stress=stress_list,
                        max_temp=tem_max,
                        max_stress=stress_max)


def load_and_interp(data_set_dir=comsol_dir):
    assert os.path.exists(data_set_dir)
    stress_ = {}
    temp_ = {}
    max_stress_ = 0.
    max_temp_ = 0.
    for fname in os.listdir(data_set_dir):
        name_tokens = re.split(r'[_.]', fname)
        idx = ''.join(name_tokens[1: -1])
        data = np.loadtxt(os.path.join(data_set_dir, fname), usecols=(0, 1, 3), dtype=np.float32)
        points = data[:, :2] * 100  # m to cm
        values = data[:, -1]
        c = griddata(points, values, (grid_x, grid_y), method='cubic', fill_value=values.mean())
        c[c < 0.] = 0.
        current_max = c.max()
        if name_tokens[0] == 'stress':
            stress_[idx] = c
            if current_max > max_stress_:
                max_stress_ = current_max
        else:
            temp_[idx] = c
            if current_max > max_temp_:
                max_temp_ = current_max

    assert stress_.keys() == temp_.keys()
    for idx in stress_.keys():
        current_stress = stress_[idx]
        current_temp = temp_[idx]
        current_stress /= max_stress_
        current_temp /= max_temp_
        current_stress[np.isnan(current_stress)] = 0.
        current_stress[current_stress < 0.] = 0.
        current_temp[np.isnan(current_temp)] = 0.
        current_temp[current_temp < 0.] = 0.

        assert current_stress.max() <= 1. and current_temp.max() <= 1., \
            'idx: {}, stress_max: {}, temp_max: {}'.format(str(idx), current_stress.max(), current_temp.max())
    return temp_, stress_,  max_temp_, max_stress_


def crop_samples(temp_, stress_, do_rotate_=False):
    vst_locate = (np.linspace(0., 1., vst_num + 2)[1: -1] * n_points).astype(int)
    radius = int(1. / (vst_num + 1) * n_points * radius_scale_factor)

    xy_mask = np.empty((2 * radius, 2 * radius) + (2,))
    xy_mask[..., 0], xy_mask[..., 1] = np.meshgrid(np.arange(2 * radius), np.arange(2 * radius))
    xy_mask -= (radius, radius)
    roi_mask = np.linalg.norm(xy_mask, axis=-1) <= radius
    angle_delta = 2 * np.pi / circular_div  # np.arctan2 \in [-pi, pi]

    _temp = []
    _stress = []
    for idx in stress_.keys():
        current_temp = temp_[idx]
        current_stress = stress_[idx]
        assert current_stress.max() <= 1. and current_temp.max() <= 1., \
            'idx: {}, stress_max: {}, temp_max: {}'.format(str(idx), current_stress.max(), current_temp.max())
        # not pythonic, but don't know how to make this better
        for x in vst_locate:
            for y in vst_locate:
                temp_roi = current_temp[x-radius: x+radius, y-radius: y+radius] * roi_mask
                stress_roi = current_stress[x-radius: x+radius, y-radius: y+radius] * roi_mask
                max_stress_ = stress_roi.max()

                # handle rotate
                if do_rotate_:
                    theta_heat = 0.
                    rotate_angle = 0.
                    for theta in range(circular_div):
                        angle_begin = theta * angle_delta - np.pi
                        angle_roi = np.arctan2(xy_mask[..., 1], xy_mask[..., 0])
                        angle_mask = np.logical_and(angle_begin <= angle_roi,  angle_roi <= angle_begin + angle_delta)
                        current_theta_heat = np.sum(temp_roi * angle_mask)
                        if current_theta_heat > theta_heat:
                            theta_heat = current_theta_heat
                            rotate_angle = angle_begin * (180 / np.pi)
                    temp_roi = rotate(temp_roi, rotate_angle, reshape=False)
                _temp.append(temp_roi)
                _stress.append(max_stress_)

    return np.array(_temp), np.array(_stress)


def generate_mx_array_itr(data_, label_, batch_size_=10):
    assert data_.shape[0] == label_.shape[0]
    n, h, w = data_.shape
    itr = mx.io.NDArrayIter(data_.reshape((n, 1, h, w)), label_, batch_size_, shuffle=True, label_name='reg_label')
    return itr


if __name__ == '__main__':
    print('{}: Loading data...'.format(datetime.now().strftime('%Y.%m.%d-%H:%M:%S')))
    tem_list, stress_list = load_and_crop()
    print('{}: Crop and interpolating data...'.format(datetime.now().strftime('%Y.%m.%d-%H:%M:%S')))
    local_interp(tem_list, stress_list)
    print('{}: Data dump done.'.format(datetime.now().strftime('%Y.%m.%d-%H:%M:%S')))
