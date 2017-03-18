# -*- coding: utf8 -*-
import mxnet as mx
import xgboost as xgb
import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage.interpolation import rotate
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import re

comsol_dir = '../Data/comsol_source'
vst_num = 12  # num of vst per axis
crop_size = 224  # follow standard CNN input size
radius_scale_factor = 0.8  # R = grid_size * radius_scale_factor
n_points = int(crop_size * (vst_num + 1) / radius_scale_factor)  # num of points interpolated per axis
grid_x, grid_y = np.mgrid[0:1:complex(0, n_points), 0:1:complex(0, n_points)]
circular_div = 16  # K in paper
feat_num = 500  # num of feature extracted per sample
plot_num = 20  # num of test samples to plot


def load_and_local_interp(data_set_dir=comsol_dir):
    assert os.path.exists(data_set_dir)
    vst_locate = np.linspace(0., 1., vst_num + 2)[1: -1]
    radius = 1. / (vst_num + 1) / 2. * radius_scale_factor
    tem_list = []
    stress_list = []
    for fname in os.listdir(data_set_dir):
        fname = os.path.splitext(fname)[0]
        tem_tokens = fname.split('_')
        if tem_tokens[0] == 'stress':
            continue
        stress_name = '_'.join(['stress'] + tem_tokens[1:]) + '.txt'
        stress_path = os.path.join(data_set_dir, stress_name)
        assert os.path.exists(stress_path)
        tem_data = np.loadtxt(os.path.join(data_set_dir, fname + '.txt'), usecols=(0, 1, 3), dtype=np.float32)
        stress_data = np.loadtxt(stress_path, usecols=(0, 1, 3), dtype=np.float32)
        tem_data[:, :2] *= 100
        stress_data[:, :2] *= 100
        for x in vst_locate:
            for y in vst_locate:
                # TODO(Rundong): not Pythonic, I gona crazy by this, believe me...
                current_tem = tem_data[
                    np.logical_and(np.logical_and((x - radius) <= tem_data[:, 0], tem_data[:, 0] <= (x + radius)),
                                   np.logical_and((y - radius) <= tem_data[:, 1], tem_data[:, 1] <= (y + radius)))]
                current_stress = stress_data[
                    np.logical_and(np.logical_and((x - radius) <= stress_data[:, 0], stress_data[:, 0] <= (x + radius)),
                                   np.logical_and((y - radius) <= stress_data[:, 1], stress_data[:, 1] <= (y + radius)))]
                stress_max = current_stress.max()
                tem_list.append(current_tem)
                stress_list.append(stress_max)

    return tem_list, stress_list


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
        c = griddata(points, values, (grid_x, grid_y), method='cubic', fill_value=0.)
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


def generate_random_feat(samples_):
    radius = int(1. / (vst_num + 1) * n_points * radius_scale_factor)
    feat_pool = np.zeros((feat_num, 2))
    for f in range(feat_num):
        current_feat = np.random.uniform(0, 2 * radius, (2,))
        while np.linalg.norm(current_feat - (radius, radius)) > radius:
            current_feat = np.random.uniform(0, 2 * radius, (2,))
        feat_pool[f] = current_feat
    feat_pool = feat_pool.astype(int)

    rand_feat = []
    for sample, max_temp_ in samples_:
        current_rand_feat = np.append(sample[feat_pool[0], feat_pool[1]], max_temp_)
        rand_feat.append(current_rand_feat)

    return np.array(rand_feat)


def xgb_regression(feat_, max_temp_):
    X = feat_[..., :-1]
    y = feat_[..., -1] * max_temp_
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    param = {'max_depth': 5, 'silent': 1, 'objective': 'reg:linear', 'eval_metric': 'rmse', 'eta': 0.02,
             'subsample': 0.8}
    watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    xgb_model = xgb.train(param, dtrain, num_boost_round=300, evals=watchlist)
    predictions = xgb_model.predict(dtest)
    actuals = y_test
    print('MSR: {}'.format(mean_squared_error(actuals, predictions)))
    plt_temp = np.random.choice(np.arange(X_test.shape[0]), plot_num, replace=False)
    plt.plot(np.arange(plot_num, dtype=int), y_test[plt_temp], 'go', predictions[plt_temp], 'rx')
    plt.show()


def generate_mx_array_itr(data_, label_, batch_size_=10):
    assert data_.shape[0] == label_.shape[0]
    n = data_.shape[0]
    h, w = data_.shape[-2:]
    itr = mx.io.NDArrayIter(data_.reshape((n, 1, h, w)), label_, batch_size_, shuffle=True, label_name='reg_label')
    return itr


if __name__ == '__main__':
    """
        We try to predict stress via random generated feature
    and xgboost.
        It's a naive struggle.
    """
    print('Loading data...')
    stress, temp, max_stress, max_temp = load_and_interp(comsol_dir)
    print('Cropping samples...')
    samples = crop_samples(stress, temp)
    print('Generating features...')
    feat = generate_random_feat(samples)
    print('Regression by XGBoost...')
    xgb_regression(feat, max_temp)
