# deep-stress
Inspired by [L Zhang, 2016](http://ieeexplore.ieee.org/document/7479196/), we try to propose a fast 3D IC stress 
estimation approach  based on convolutional Neural Network.

This net can work now, please run by `python tools/predict_via_mx.py` in this __root__ folder.

# Hyper-param configurations

We have tried some of the exsist net structure which been proved work well in computer vision tasks,
with their last output layer modifie into two-stacked fully connected layer with one output (*see
`build_Nets.py` for more detail*).

Bellow are some of the hyper-params which can work on these nets. Module files will be relseased soon.
(in fact training them is fast... this nets're small enough to be trained in minutes)

## VGG-Like-11
| base_lr | rescan_epoch | rescan_factor | batch_size | num_epoch | RMSE |
|:----:|:----:|:----:|:----:|:----:|:----:|
| 0.03 | 250 | 0.9 | 256 | 100 | 0.03135 |

## LeNet
| base_lr | rescan_epoch | rescan_factor | batch_size | num_epoch | RMSE |
|:----:|:----:|:----:|:----:|:----:|:----:|
| 0.1 | 250 | 0.9 | 10 | 10 | 0.02993 |

# B.D.Thesis Draft
[Avaliable here](https://github.com/CrazyRundong/UESTCthesis), still struggling...
