# -*- coding: utf-8 -*-
import src.prepare_data as Data
import src.build_Nets as Net

import mxnet as mx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

import os
import logging
logging.getLogger().setLevel(logging.DEBUG)


def main():
    batch_size = 10
    momentum = 0.9
    num_epoch = 5
    val_ratio = 0.3
    disp_batch = 50
    num_batch = 5

    base_lr = 0.1
    lr_rescan_epoch = 500
    lr_rescan_factor = 0.9

    data_path = '../Data/comsol_crops.npz'

    if not os.path.exists(data_path):
        Data.main()
    data_set = np.load(data_path)
    x = data_set['temp']
    y = data_set['stress']
    max_temp = data_set['max_temp']
    max_stress = data_set['max_stress']
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_ratio, random_state=0)

    train_iter = Data.generate_mx_array_itr(x_train, y_train, batch_size)
    val_iter = Data.generate_mx_array_itr(x_val, y_val, batch_size, shuffle_=False)
    net = Net.build_lenet()
    # net = Net.build_tiny_yolo()
    metric = mx.metric.RMSE()
    lr_sch = mx.lr_scheduler.FactorScheduler(step=lr_rescan_epoch, factor=lr_rescan_factor)

    model = mx.mod.Module(
        context=mx.gpu(0),
        symbol=net,
        data_names=['data'],
        label_names=['reg_label']
    )
    # model.bind(data_shapes=train_iter.provide_data,
    #            label_shapes=train_iter.provide_label)
    # model.init_params(mx.initializer.Normal())
    model.fit(
        train_iter,
        eval_data=val_iter,
        optimizer='sgd',
        optimizer_params={'learning_rate': base_lr, 'momentum': momentum, 'lr_scheduler': lr_sch},
        eval_metric=metric,
        num_epoch=num_epoch,
        batch_end_callback=mx.callback.Speedometer(batch_size, disp_batch)
    )
    score = model.score(val_iter, metric)
    for idx_, score_ in score:
        print('Module {} score: {}'.format(idx_, score_))

    gt_stress = y_val[:batch_size * num_batch]
    predict_stress = model.predict(val_iter, num_batch)
    mse = mean_squared_error(gt_stress * max_stress, predict_stress.asnumpy() * max_stress)
    print("MSE: {}".format(mse))
    plt.plot(range(batch_size * num_batch),
             gt_stress * max_stress, 'go',
             predict_stress.asnumpy() * max_stress, 'rx')
    plt.show()

if __name__ == '__main__':
    main()
