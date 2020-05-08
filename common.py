from Abstract import AbstractSimilarity, AbstractDistance
import numpy as np


class ImageFeature(object):
    def __init__(self, img_name, feature):
        # self._item_id = item_id
        self._img_name = img_name
        # self._item_box = item_box
        self._feature = feature
        # self._classify = classify

    # @property
    # def item_id(self):
    #     """
    #     id,也就是文件夹名称
    #     """
    #     return self._item_id
    #
    # @item_id.setter
    # def item_id(self, value):
    #     self._item_id = str(value)

    @property
    def img_name(self):
        return self._img_name

    @img_name.setter
    def img_name(self, value):
        self._img_name = str(value)

    @property
    def feature(self):
        """
        物品的特征向量表示
        :return:
        """
        return self._feature

    @feature.setter
    def feature(self, value):
        # if not isinstance(value, np.ndarray):
        #     value = np.array(value, shape=(-1,), dtype=np.float)
        self._feature = value

    # @property
    # def classify(self):
    #     return self._classify
    #
    # @classify.setter
    # def classify(self, value):
    #     try:
    #         self._classify = int(value)
    #     except:
    #         self._classify = -1


class CosSimilarity(AbstractSimilarity):

    @staticmethod
    def get_similarity(f1, f2):
        if isinstance(f1, list):
            f1 = np.array(f1).reshape((-1,))
        if isinstance(f2, list):
            f2 = np.array(f2).reshape((-1,))
        return CosSimilarity.get_cos(f1, f2)

    @staticmethod
    def get_cos(vector_a, vector_b):
        vector_a = np.reshape(vector_a, (-1))
        vector_b = np.reshape(vector_b, (-1))
        num = np.dot(vector_a, vector_b.T)  # 若为行向量则 A * B.T
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        if denom == 0:
            sim = 0.0
        else:
            cos = num / denom  # 余弦值
            sim = 0.5 + 0.5 * cos  # 归一化
        return sim


class Euclidean_Distance(AbstractDistance):
    """欧氏距离"""

    @staticmethod
    def get_distance(feature1, feature2, flags=''):
        distance = np.power(feature1 - feature2, 2).sum()
        return distance
