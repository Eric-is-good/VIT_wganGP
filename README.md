# VIT_wganGP


## 一个结合了 vision transformer 和 wgangp 的对抗生成网络，生成64x64 动漫头像，同时提供一个传统卷积网络的训练好的模型，仅仅 20 M.



## 新增了 VAE 模型，效果要好一点，并提供预训练模型（30M）

## 示例

![recons_7](https://github.com/Eric-is-good/VIT_wganGP/blob/main/out/recons_7.png)







## 模型切换



### 我们提供了两个模型（默认是  反卷积）

1. 基于 vision transformer 的生成网络和判别网络。
2. 基于  反卷积  的生成网络和判别网络。



### 如何切换

1. 进入 w_gp_gan.py 文件，将 下图的 covg 和 covd 更换成 vitg 和 vitd 即可。



1. ![](https://github.com/Eric-is-good/VIT_wganGP/blob/main/readmepic/1.jpg)

  

​     2.  vision transformer 的 z_dim 是 1024，也需要修改。

![2](https://github.com/Eric-is-good/VIT_wganGP/blob/main/readmepic/2.jpg)





## 测试与训练

1. 测试使用  generate_img.py ，不需要显卡。
2. 训练使用  w_gp_gan.py ，需要显卡。



## 数据集

1. 直接把图片放在 data 下，运行 utils 的 read_path 自动转化为 64x64 的图片大小。



## 模型

提供一个反卷积的模型，仅仅 20 M.
