# -*- coding: utf8 -*-
import src.prepare_data as Data
import src.build_Nets as Net

import mxnet as mx
import numpy as np
from sklearn.model_selection import train_test_split

import logging
logging.getLogger().setLevel(logging.DEBUG)


def main():
    batch_size = 10
    base_lr = 0.1
    momentum = 0.9
    num_epoch = 10
    val_ratio = 0.3
    disp_batch = 50
    data_path = '../Data/consol_crops.npz'

    data_set = np.load(data_path)
    x = data_set['temp']
    y = data_set['stress']
    max_temp = data_set['max_temp']
    max_stress = data_set['max_stress']
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_ratio, random_state=0)

    train_iter = Data.generate_mx_array_itr(x_train, y_train, batch_size)
    val_iter = Data.generate_mx_array_itr(x_val, y_val, batch_size)
    lenet = Net.build_lenet()
    metric = mx.metric.RMSE()
    model = mx.mod.Module(
        context=mx.gpu(0),
        symbol=lenet,
        data_names=['data'],
        label_names=['reg_label']
    )
    model.init_params(mx.initializer.Xavier())
    model.fit(
        train_iter,
        eval_data=val_iter,
        optimizer='sgd',
        optimizer_params={'learning_rate': base_lr, 'momentum': momentum},
        eval_metric=metric,
        num_epoch=num_epoch,
        batch_end_callback=mx.callback.Speedometer(batch_size, disp_batch)
    )
    score = model.score(val_iter, metric)
    for idx_, score_ in score:
        print('Module {} score: {}'.format(idx_, score_))

if __name__ == '__main__':
    main()
