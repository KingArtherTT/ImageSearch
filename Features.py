import cv2
from Abstract import AbstractFeature
import numpy as np


class SURF(AbstractFeature):
    def __init__(self, cluster_center=500):
        self.surf = cv2.xfeatures2d.SURF_create(
            hessianThreshold=3000  # 默认100，关键点检测的阈值，越高监测的点越少
            , nOctaves=4  # 默认4，金字塔组数
            , nOctaveLayers=3  # 默认3，每组金子塔的层数
            , extended=True  # 默认False，扩展描述符标志，True表示使用扩展的128个元素描述符，False表示使用64个元素描述符
            , upright=False  # 默认False，垂直向上或旋转的特征标志，True表示不计算特征的方向，False-计算方向。
        )
        self.cluster_center = cluster_center

    def get_surf(self, filename, box=None):
        img = cv2.imread(filename)  # 读取文件
        if box is not None:
            img = img[box[1]:box[3], box[0]:box[2], :]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转化为灰度图
        # surf = cv2.xfeatures2d.SURF_create()
        keyPoint, descriptor = self.surf.detectAndCompute(img, None)  # 特征提取得到关键点以及对应的描述符（特征向量）
        return img, keyPoint, descriptor

    def set_kmeans_model(self, kmeans):
        self.kmeans = kmeans

    def get_feature(self, img, bboxes=None):
        if self.kmeans is None:
            raise ValueError('需要先给 self.kmeans 赋值')
        img, keyPoint, descriptor = self.get_surf(img, bboxes)
        feature = np.zeros(self.cluster_center, dtype=np.float)
        if descriptor is not None:
            cluster_classify = self.kmeans.predict(descriptor)
            for i in range(len(cluster_classify)):
                score = self.kmeans.score(np.reshape(descriptor[i], (1, -1)) )# 等价于distance 不过带有方向
                feature[int(cluster_classify[i])] += np.abs(score)

        return feature
