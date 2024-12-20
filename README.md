# ![Overview of the UniHENN](./figure/Overview.png)

Official repository for **UniHENN: Designing More Versatile Homomorphic Encryption-based CNNs without im2col**  
by
Hyunmin Choi<sup>1,2</sup>, Jihun Kim<sup>2</sup>, Seungho Kim<sup>2</sup>, Seonhye Park<sup>2</sup>, Jeongyong Park<sup>2,3</sup>, Wonbin Choi<sup>1</sup>, and Hyoungshick Kim<sup>2**</sup>

<sup>1</sup> NAVER Cloud, South Korea
<sup>2</sup> Sungkyunkwan University, South Korea
<sup>3</sup> Samsung Electronics, South Korea

** Corresponding author

You can read the paper here: [ArXiv Link](https://arxiv.org/abs/2402.03060) [IEEE ACCESS](https://ieeexplore.ieee.org/document/10623483)

## Designing Faster and More Versatile Homomorpic Encryption-based CNNs
UniHENN is a Python package that provides Homomorphic Encryption(HE)-based CNN inference. 
It provides the following features:
- HE-based CNN inference compatible with pytorch library
  - Automatically convert each layer to a compatible HE-based layer
  - Not only offers 2D-CNN, but also offers 1D CNN inference.
- It offers batch operation to optimize the inference time

## Server Setting

-   We use NAVER Cloud servers (https://www.ncloud.com/product/compute/server)
-   In this scenario, we use the following server spec.
    -   All server spec are Standard-g2 Server.
    -   CentOS 7.8.64
    -   16 cores (Intel(R) Xeon(R) Gold 5220 CPU @ 2.20GHz) with 64GB DRAM

## Requirements

-   Python 3.8.1
-   SEAL-Python (Release 4.0.0, https://github.com/Huelse/SEAL-Python)
-   Pytorch
-   TorchVision
-   numpy
-   pandas

## How to use

If you want to test the M3 model, enter the following command in the shell:
```
python main.py 3
```
You can enter values from 1 to 6.

## Dataset

We use MNIST and CIFAR10 datasets for 2D CNN models which can be downloaded using Torchvision. The details about datasets can be found at https://pytorch.org/vision/0.15/datasets.html.

## HE Parameters
We use the Cheon-Kim-Kim-Song(CKKS) scheme following parameter settings:
|Parameter|Value|
|---|---|
|\# of slots|8,192|
|log Q|432|
|PK(MB)|1.87|
|SK(MB)|0.94|
|GK(GB)|0.57|
|RK(MB)|22.52|
|ctxt(MB)|1.68|
|\# of mult(depth)|11|

The PK, SK, GK, and RK mean the public key, secret key, galois key, and relinearization key each other.
These parameter settings guarantee the 128-bit security parameters.


## Models
We conducted seven experiments to test the performance of UniHENN with various models and data. The test file is in the uni_henn directory.

|Model|Type|Dataset|\# of layers|Test file|
|---|---|---|---|---|
|M1|2D CNN|MNIST|6|M1_test.py|
|M2|2D CNN|MNIST|8|M2_test.py|
|M3|2D CNN|MNIST|7|M3_test.py|
|M4|2D CNN|MNIST|12|M4_test.py|
|M5|2D CNN|Cifar-10|11|M5_test.py|
|M6|1D CNN|ECG Dataset|7|M6_test.py|

## Contributing
- Ownership: Hyunmin Choi
- Architecture Designer: Hyunmin Choi(https://github.com/hm-choi)
- Main Contributor: Jihun Kim (https://github.com/JihunSKKU), Hyunmin Choi(https://github.com/hm-choi)
- Contributor: Seungho Kim (https://github.com/Seungho-Kim-SKKU)
- Code Reviewer: Wonbin Choi (https://github.com/bindon)
- Register issues and pull requests are welcome. If there are some errors or changes then please open an issue and write down the details. 

## Publications
UniHENN: Designing Faster and More Versatile Homomorphic Encryption-based CNNs without im2col. (IEEE ACCESS, 2024)

## Citations
```
@article{choi2024unihenn,
  title={UniHENN: Designing Faster and More Versatile Homomorphic Encryption-based CNNs without im2col},
  author={Choi, Hyunmin and Kim, Jihun and Kim, Seungho and Park, Seonhye and Park, Jeongyong and Choi, Wonbin and Kim, Hyoungshick},
  journal={IEEE Access},
  year={2024},
  publisher={IEEE}
}
```
## License
This is available for the non-commercial purpose only. 
