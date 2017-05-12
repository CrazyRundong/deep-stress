# -*- coding: utf-8 -*-
import mxnet as mx


# LeNet
def build_lenet():
    data = mx.symbol.Variable('data')
    label = mx.symbol.Variable('reg_label')
    # first conv layer
    conv1 = mx.sym.Convolution(data=data, kernel=(5, 5), num_filter=20)
    tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")  # tanh
    pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2, 2), stride=(2, 2))
    # second conv layer
    conv2 = mx.sym.Convolution(data=pool1, kernel=(5, 5), num_filter=50)
    tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
    pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2, 2), stride=(2, 2))
    # first fullc layer
    flatten = mx.sym.Flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=25)  # origin lenet get 500 nodes
    tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
    # second fullc
    fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=1)
    # softmax loss
    # lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')
    lenet = mx.sym.LinearRegressionOutput(data=fc2, label=label, name='linear_reg')

    return lenet


# Dark Net in YOLO 2000
def build_conv_block(**kwargs):
    conv = mx.sym.Convolution(**kwargs)
    bn = mx.sym.BatchNorm(data=conv, fix_gamma=False, eps=1e-5 + 1e-10, momentum=0.9,)
    lkReLU = mx.sym.LeakyReLU(data=bn)
    # lkReLU = mx.sym.Activation(data=bn, act_type="relu")

    return lkReLU


def build_conv_pool_block(*args, **kwargs):
    n_conv, n_pool = args
    data = kwargs['data']
    for i in range(n_conv):
        data = build_conv_block(data=data, **kwargs['conv{}'.format(i)])
    if n_pool:
        pool = mx.sym.Pooling(data=data, **kwargs['pool'])
    else:
        pool = data

    return pool


def build_tiny_yolo():
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('reg_label')
    pool1 = build_conv_pool_block(4, 1,
                                  data=data,
                                  conv0={'kernel': (1, 1), 'stride': (1, 1), 'pad': (0, 0), 'num_filter': 16},
                                  conv1={'kernel': (3, 3), 'stride': (1, 1), 'pad': (1, 1), 'num_filter': 32},
                                  conv2={'kernel': (1, 1), 'stride': (1, 1), 'pad': (0, 0), 'num_filter': 16},
                                  conv3={'kernel': (3, 3), 'stride': (1, 1), 'pad': (1, 1), 'num_filter': 32},
                                  pool={'kernel': (2, 2), 'stride': (2, 2), 'pool_type': 'max'})
    pool2 = build_conv_pool_block(4, 1,
                                  data=pool1,
                                  conv0={'kernel': (1, 1), 'stride': (1, 1), 'pad': (0, 0), 'num_filter': 32},
                                  conv1={'kernel': (3, 3), 'stride': (1, 1), 'pad': (1, 1), 'num_filter': 256},
                                  conv2={'kernel': (1, 1), 'stride': (1, 1), 'pad': (0, 0), 'num_filter': 32},
                                  conv3={'kernel': (3, 3), 'stride': (1, 1), 'pad': (1, 1), 'num_filter': 256},
                                  pool={'kernel': (2, 2), 'stride': (2, 2), 'pool_type': 'avg'})
    flatten = mx.sym.Flatten(data=pool2)
    # fc1 = mx.symbol.FullyConnected(data=pool2, num_hidden=50)
    # fc1_lkReLU = mx.sym.LeakyReLU(data=fc1)
    fc2 = mx.symbol.FullyConnected(data=flatten, num_hidden=1)
    yolo = mx.sym.LinearRegressionOutput(data=fc2, label=label, name='linear_reg')

    return yolo


# Our New Net
def build_net_a():
    """ based on the insight of VGG Net:
    2 * Conv((3, 3), stride=1, pad=1) == Conv((5, 5), stride=1, pad=1)
    n_channel *= 2 after each pooling layer
    we follow the channel num of LeNet
    """
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('reg_label')
    pool1 = build_conv_pool_block(2, 1,
                                  data=data,
                                  conv0={'kernel': (3, 3), 'stride': (1, 1), 'pad': (1, 1), 'num_filter': 20},
                                  conv1={'kernel': (1, 1), 'stride': (1, 1), 'pad': (0, 0), 'num_filter': 40},
                                  conv2={'kernel': (3, 3), 'stride': (1, 1), 'pad': (1, 1), 'num_filter': 20},
                                  pool={'kernel': (2, 2), 'stride': (2, 2), 'pool_type': 'max'})
    pool2 = build_conv_pool_block(2, 1,
                                  data=pool1,
                                  conv0={'kernel': (3, 3), 'stride': (1, 1), 'pad': (1, 1), 'num_filter': 40},
                                  conv1={'kernel': (3, 3), 'stride': (1, 1), 'pad': (1, 1), 'num_filter': 40},
                                  pool={'kernel': (2, 2), 'stride': (2, 2), 'pool_type': 'max'})
    pool3 = build_conv_pool_block(2, 1,
                                  data=pool2,
                                  conv0={'kernel': (3, 3), 'stride': (1, 1), 'pad': (1, 1), 'num_filter': 80},
                                  conv1={'kernel': (3, 3), 'stride': (1, 1), 'pad': (1, 1), 'num_filter': 80},
                                  pool={'kernel': (2, 2), 'stride': (2, 2), 'pool_type': 'max'})
    pool4 = build_conv_pool_block(2, 1,
                                  data=pool3,
                                  conv0={'kernel': (3, 3), 'stride': (1, 1), 'pad': (1, 1), 'num_filter': 160},
                                  conv1={'kernel': (3, 3), 'stride': (1, 1), 'pad': (1, 1), 'num_filter': 160},
                                  pool={'kernel': (2, 2), 'stride': (2, 2), 'pool_type': 'max'})
    flatten = mx.sym.Flatten(data=pool4)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=50)  # origin lenet get 500 nodes
    tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
    # second fullc
    fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=1)
    # softmax loss
    net_a = mx.sym.LinearRegressionOutput(data=fc2, label=label, name='linear_reg')

    return net_a
