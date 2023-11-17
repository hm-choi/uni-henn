# Designing Faster and More Versatile Homomorpic Encryption-based CNNs

## Server Setting

-   We use NAVER Cloud servers (https://www.ncloud.com/product/compute/server)
-   In this scenario, we use the following server spec.
    -   All server spec are Standard-g2 Server.
    -   CentOS 7.8.64
    -   16 cores Intel(R) Xeon(R) Gold 5220 CPU @ 2.20GHz) with 64GB DRAM

## Requirements

-   Python 3.8.1
-   SEAL-Python (Release 4.0.0, https://github.com/Huelse/SEAL-Python)
-   Pytorch
-   TorchVision
-   numpy
-   pandas

## How to use

```
python M*_test.py
```

## Dataset

We use MNIST, CIFAR10, and USPS datasets for 2D CNN models which can be downloaded using Torchvision. The details about datasets can be found in https://pytorch.org/vision/0.15/datasets.html.
