# DTT

> [Deep Tri-Training for Semi-Supervised Image Segmentation](https://ieeexplore.ieee.org/document/9804753)
>
> by Shan An; Haogang Zhu; Jiaao Zhang; Junjie Ye; Siliang Wang; Jianqin Yin; Hong Zhang

## Installation
The code is developed using Python 3.6 with PyTorch 1.0.0. The code is developed and tested using 4 or 8 Tesla V100 GPUs.

1. **Clone this repo.**

   ```shell
   $ git clone https://github.com/anshan-ar/DTT.git
   $ cd DTT
   ```

2. **Install dependencies.**

   **(1) Create a conda environment:**

   ```shell
   $ conda env create -f DTT.yaml
   $ conda activate DTT
   ```

   **(2) Install apex 0.1(needs CUDA)**

   ```shell
   $ cd ./furnace/apex
   $ python setup.py install --cpp_ext --cuda_ext
   ```

## Getting Started
### Data Preparation 
##### Download the data (PASCAL VOC, Cityscapes)
- [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
- [Cityscapes](https://www.cityscapes-dataset.com/)

### Training && Inference on PASCAL VOC:
   ```shell
   $ cd ./voc8.res50v3+.DTT
   $ bash script.sh
   ```
![method](https://github.com/anshan-ar/DTT/blob/main/method.png)
- Base model obtained by CPS/CPS+CutMix training.
- The third network branch is obtained by DTT training on the basis of the model obtained above.
- Final results are obtained through the DTT voting mechanism.

### Different Partitions
To try other data partitions beside 1/8, you just need to change two variables in `config.py`:
```python
C.labeled_ratio = 8
C.nepochs = 34
```
Please note that, for fair comparison, we control the total iterations during training in each experiment similar (almost the same), including the supervised baseline and semi-supervised methods. Therefore, the nepochs for different partitions are different. 

We take VOC as an example.
1. We totally have 10582 images. The full supervised baseline is trained for 60 epochs with batch size 16, thus having 10582*60/16 = 39682.5 iters.
2. If we train CPS under the 1/8 split, we have 1323 labeled images and 9259 unlabeled images. Since the number of unlabeled images is larger than the number of labeled images, the `epoch` is defined as passing all the unlabeled images to the network. In each iteration, we have 8 labeled images and 8 unlabeled images, thus having 9259/8 = 1157.375 iters in one epoch. Then the total epochs we need is 39682.5/1157.375 = 34.29 ≈ 34. 
3. For the supervised baseline under the 1/8 split, the batch size 8 and the iteration number is 39682.5 (the same as semi-supervised method).


We list the nepochs for different datasets and partitions in the below.

| Dataset    | 1/16 | 1/8  | 1/4  | 1/2  |
| ---------- | ---- | ---- | ---- | ---- |
| VOC        | 32   | 34   | 40   | 60   |
| Cityscapes | 128  | 137  | 160  | 240  |
## Citation

Please consider citing this project in your publications if it helps your research.

```bibtex
@ARTICLE{9804753,
  author={An, Shan and Zhu, Haogang and Zhang, Jiaao and Ye, Junjie and Wang, Siliang and Yin, Jianqin and Zhang, Hong},
  journal={IEEE Robotics and Automation Letters}, 
  title={Deep Tri-Training for Semi-Supervised Image Segmentation}, 
  year={2022},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/LRA.2022.3185768}}
```
