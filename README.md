# VIT_wganGP


## 一个结合了 vision transformer 和 wgangp 的对抗生成网络，生成64x64 动漫头像，同时提供一个传统卷积网络的训练好的模型，仅仅 20 M.



## 新增了 VAE 模型，效果要好一点，并提供预训练模型（30M）

## wgan 生成示例

![recons_7](https://github.com/Eric-is-good/VIT_wganGP/blob/main/out/recons_7.png)


## vae 生成示例

![recons_7](https://github.com/Eric-is-good/VIT_wganGP/blob/main/readmepic/true_1.png)



## 测试与训练

1. 测试使用  generate_img.py ，不需要显卡。
2. 训练使用  train_for_wgangp.py 或 train_for_vae.py，需要显卡。




## 模型切换

### 我们提供了三个模型（默认是  反卷积 + wgangp）

1. 基于 vision transformer 的生成网络和判别网络。（vit + wgangp）
2. 基于  反卷积 (cov) 的生成网络和判别网络。（cov + wgangp）
2. 基于  反卷积 (cov) 的生成网络和判别网络。（cov + vae）



### 如何切换

进入 train_for_wgangp.py 文件，在 main 里面，通过 model = WGPGAN(model_name="cov")  的 model_name 切换 vit 和 cov，使用 wgangp。

进入 train_for_vae.py 文件使用 vae。



## 数据集

直接把图片放在 raw_pics 下，自动转化为 64x64 的图片大小，并保存在pics下 。就可以训练了。



## 模型

提供一个反卷积的模型，仅仅 20 M，在 model 里面，cov.pt 是 cov 的模型。

vae 的模型由于大于 20 MB，被压缩成了两个文件，使用前先解压 vae.zip。

