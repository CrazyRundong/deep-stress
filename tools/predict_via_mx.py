# -*- coding: utf-8 -*-
import src.prepare_data as Data
import src.build_Nets as Net

import mxnet as mx
import numpy as np
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

import os
import logging
logging.getLogger().setLevel(logging.DEBUG)


def main(draw_net=False):
    batch_size = 256
    momentum = 0.9
    num_epoch = 100
    disp_batch = 50
    num_batch = 1
    num_plot = 25

    base_lr = 0.01
    lr_rescan_epoch = 250
    lr_rescan_factor = 0.9

    data_path = './Data/comsol_crops.npz'

    if not os.path.exists(data_path):
        Data.main()
    data_set = np.load(data_path)
    x_train = data_set['temp_train']
    x_val = data_set['temp_val']
    y_train = data_set['stress_train']
    y_val = data_set['stress_val']
    max_temp = data_set['max_temp']
    max_stress = data_set['max_stress']

    train_iter = Data.generate_mx_array_itr(x_train, y_train, batch_size)
    val_iter = Data.generate_mx_array_itr(x_val, y_val, batch_size, shuffle_=False)

    # net = Net.build_lenet()
    net = Net.build_net_a()

    # show model
    if draw_net:
        g = mx.viz.plot_network(net, shape={'data': (batch_size, 1,) + x_train.shape[-2:], 'reg_label': (batch_size,)})
        g.render()

    metric = mx.metric.RMSE()
    lr_sch = mx.lr_scheduler.FactorScheduler(step=lr_rescan_epoch, factor=lr_rescan_factor)

    model = mx.mod.Module(
        context=mx.gpu(0),
        symbol=net,
        data_names=['data'],
        label_names=['reg_label']
    )
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

    plot_idx = np.random.choice(batch_size * num_batch, num_plot, replace=False)
    predict_stress = model.predict(val_iter, num_batch)
    gt_stress = y_val[plot_idx]
    predict_stress = predict_stress.asnumpy()[plot_idx]

    mse = mean_squared_error(gt_stress * max_stress, predict_stress * max_stress)
    print("MSE: {}".format(mse))
    plt.plot(range(num_plot),
             gt_stress * max_stress, 'go',
             predict_stress * max_stress, 'rx')
    plt.show()

if __name__ == '__main__':
    main()
