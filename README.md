# ImageSearch
使用SURF+Kmeans建立的图像检索系统（CBIR）

### 系统实验数据集说明

本系统是基于内容的图像检索系统，实验数据集是“The Oxford Buildings Dataset”，数据集介绍链接：http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/ ；数据集下载链接：http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz

### 系统匹配度说明

若采用数据集中标注为“good”的图片进行查询，查询结果的Top-4可以完全匹配标注数据（即Top-4是标注数据的一个子集）。



若采用数据集中标注为“ok”的图片进行查询，查询结果的Top-4可以部分匹配标注数据，4张照片中偶尔会出现不是标注数据的照片。以4张照片若出现不是标注数据的照片为错误匹配来计算，匹配率大概50%。



若采用数据集中标注为“junk”的图片进行查询，查询结果的Top-4可以极少的匹配标注数据，4张照片中至少有一张属于标注数据的概率大概在60%。

### 系统主体算法说明

本系统主要解决的是无标签的图像检索问题，经过一系列处理后，最终将每张图片表示成一个500纬的向量，并利用有归一化的余弦距离度量不同向量的相似度，其中0为完全不相似，1为完全相似。由此来度量用户每次输入的图片，并输出相似度排名最高的四张图片。

1. 随机抽样总体数据集的十分之一，用于SURF特征点检测，调整SURF的参数，使得单个特征点的描述向量为128纬，向量个数大概在300K左右，保存为特征点描述数据集：all_descriptor.pkl
2. 使用k-means算法进行聚类分析，将所有特征点聚类为500类，形成聚类模型：K-MEANS-500；（聚类过程中由于计算量过于庞大，因此也采用了 MiniBatchKMeans 用于加速聚类）
3. 建立图像检索的图片特征库，记作all_img_features：依次输入总体数据集，每张图片都要经过第四步的处理，并得到一个500纬的向量用于表示图片。
4. 图片先经过SURF特征点检测，得到若干 keypoints 和相应的 descriptors ，随后将 descriptors 输入聚类模型： K-MEANS-500，得到每个descriptor所属类别 A，以及与类别中心的距离 S；随后初始化图像的特征向量Vector为全0向量，接着将Vector[A]处的值与S进行累加；逐一处理了所有的 descriptors后，即可得到图像的特征向量表示Vector，并输出Vector。
5. 将用户输入的图片也进行第四步的处理，得到该图像的特征向量表示,，记作Anchor.
6. 使用有归一化的余弦距离度量Anchor与all_img_features，输出其中相似度最高的四张图片（需要剔除Anchor自身）。



### 系统效果演示

![](pictures/系统演示.gif)



### 系统使用步骤

1. 环境要求：

需安装以下python库：

```
python==3.7
opencv-contrib-python ==3.4.2.17
scikit-learn==0.22.1
scipy==1.4.1
Flask==1.1.2
```

2. 需要下载数据集“The Oxford Buildings Dataset”，下载链接：http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz

   下载完成并解压缩后，将数据集的**oxbuild_images** 文件夹整体复制到项目的 **/static/img_data/** 路径下。

   若采用其它数据集，请注意修改文件夹名称，以及代码中的相应路径。

3. 训练K-Means模型，详情请查看 ClusteringAnalysis.py 文件。

   根据数据集所在文件夹的不同，需要修改路径参数；

   根据训练数据集大小的不同，可能需要修改KMeans或SURF的参数；

   若数据集过大，请考虑使用MiniBatchKMeans

4. 建立图片特征库，详情请查看 BuildImageFeature.py 文件。

   根据数据集所在文件夹的不同，需要修改路径参数。

5. 用FLASK，cd到项目根目录，进行如下命令行操作（以Ubuntu为例）：

```shell
 # 非调试模式：
export FLASK_APP=start.py
flask run

# 调试模型：
export FLASK_APP=start.py
export FLASK_ENV=development
flask run
```

