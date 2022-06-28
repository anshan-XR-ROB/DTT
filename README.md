# DTT

> [Deep Tri-Training for Semi-Supervised Image Segmentation](https://ieeexplore.ieee.org/document/9804753)
>
> by Shan An; Haogang Zhu; Jiaao Zhang; Junjie Ye; Siliang Wang; Jianqin Yin; Hong Zhang


## Datasets

##### Download the data (PASCAL VOC, Cityscapes)
- [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
- [Cityscapes](https://www.cityscapes-dataset.com/)

### Training on PASCAL VOC:
   ```shell
   $ cd ./voc8.res50v3+.DTT
   $ bash script.sh
   ```
![method](https://github.com/anshan-ar/DTT/blob/main/method.png)
- Base model obtained by CPS/CPS+CutMix training.
- The third network branch is obtained by DTT training on the basis of the model obtained above.
- Final results are obtained through the DTT voting mechanism.


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
