# DGCNN-Pytorch
A Re-Implement of Dynamic Graph CNN for Point-Cloud Classification and Segmentation



# Dynamic Graph CNN for Learning on Point Clouds

We propose a new neural network module dubbed EdgeConv suitable for CNN-based high-level tasks on point clouds including classification and segmentation. EdgeConv is differentiable and can be plugged into existing architectures.

[[Project\]](https://liuziwei7.github.io/projects/DGCNN) [[Paper\]](https://arxiv.org/abs/1801.07829)

## Overview

`DGCNN-Pytorch` is my personal re-implementation of Dynamic Graph CNN.
## Run
### Point-Cloud Data Preparations
There is two ways to convert ModelNet40 PLY or OFF file to Point-Cloud.
1. Use `h5_dataloader.py` download and `load modelnet40_ply_hdf5_2048` files

2. Custom down-sampling points from mesh. Download Modelnet40 off file, and unzip it in `Data/ModelNet40`
  Run `Sampler` with `test = 0` and `test = 1`, and sampled point-cloud file will save in `ModelNet40_`
  Next  run `pointcloud_dataloader` to convert `*.points` to h5 file. `Data\ModelNet40_`  folder will create `ModelNet40_test.h5` and `ModelNet40_train.h5`

  

## Train

Train model: Run `train` to train your model. Now is PointNet, next will update to pointnet and DGCNN.



## To-Do

Next few days, will upload DGCNN model.

## Citation

Please cite this paper if you want to use it in your work,

```
@article{dgcnn,
  title={Dynamic Graph CNN for Learning on Point Clouds},
  author={Wang, Yue and Sun, Yongbin and Liu, Ziwei and Sarma, Sanjay E. and Bronstein, Michael M. and Solomon, Justin M.},
  journal={ACM Transactions on Graphics (TOG)},
  year={2019}
}
```

## License

MIT License