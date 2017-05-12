# -*- coding: utf-8 -*-
import mxnet as mx
import numpy as np
from scipy.interpolate import griddata
from datetime import datetime
import os
from sklearn.model_selection import train_test_split

comsol_dir = './Data/comsol_source'
data_path = './Data/comsol_crops.npz'
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
        tem[:, :2] += 0.5  # norm (x, y) to [0, 1]
        c = griddata(tem[:, :2], tem[:, -1], (xx, yy), method='cubic', fill_value=tem[:, -1].mean())
        interped_list.append(c)
    interped_list = np.array(interped_list)
    stress_list = np.array(stress_list)

    # Train / Val split
    temp_train, temp_val, stress_train, stress_val = train_test_split(interped_list, stress_list, train_size=0.7)

    tem_mean = np.mean(temp_train, axis=0)
    tem_var = temp_train.var()

    temp_train -= tem_mean
    temp_val -= tem_mean
    temp_train /= tem_var
    temp_val /= tem_var

    tem_max = np.abs(temp_train).max()

    stress_max = stress_train.max()
    stress_train /= stress_max
    stress_val /= stress_max

    # Dump npz
    np.savez_compressed(npz_dir,
                        temp_train=temp_train,
                        temp_val=temp_val,
                        stress_train=stress_train,
                        stress_val=stress_val,
                        max_temp=tem_max,
                        max_stress=stress_max)


def generate_mx_array_itr(data_, label_, batch_size_=10, shuffle_=True):
    assert data_.shape[0] == label_.shape[0]
    n, h, w = data_.shape
    itr = mx.io.NDArrayIter(data_.reshape((n, 1, h, w)), label_, batch_size_, shuffle=shuffle_, label_name='reg_label')
    return itr


def main():
    print('{}: Loading data...'.format(datetime.now().strftime('%Y.%m.%d-%H:%M:%S')))
    tem_list, stress_list = load_and_crop()
    print('{}: Crop and interpolating data...'.format(datetime.now().strftime('%Y.%m.%d-%H:%M:%S')))
    local_interp(tem_list, stress_list)
    print('{}: Data dump done.'.format(datetime.now().strftime('%Y.%m.%d-%H:%M:%S')))

if __name__ == '__main__':
    main()
