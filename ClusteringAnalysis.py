import os
import numpy as np
import cv2
import time
from Features import SURF
from sklearn.cluster import KMeans, MiniBatchKMeans
import joblib
import pickle
import random


# KMeans 和 MiniBatchKMeans 区别何在

def build_sample_center(paths, centers=500):
    """建立样本中心"""
    # 1.获取所有图片的SURF特征点描述
    # 2.根据特征点描述向量进行聚类，获得样本中心
    # 3.返回样本中心
    if not isinstance(paths, (tuple, list)):
        paths = [paths]
    all_file_path = []
    for p in paths:
        filenames = os.listdir(p)
        all_file_path.extend([p + f for f in filenames])
    random.shuffle(all_file_path)
    # all_file_path = all_file_path[:500]
    print('{} 共计图片数目:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), len(all_file_path)))
    if os.path.exists('./pickle/all_descriptor.pkl'):
        with open('./pickle/all_descriptor.pkl', 'rb') as f:
            all_descriptor = pickle.load(f)
    else:
        all_descriptor = get_surf_descriptor(all_file_path)
        with open('./pickle/all_descriptor.pkl', 'wb') as f:
            pickle.dump(all_descriptor, f)
    random.shuffle(all_descriptor)
    all_descriptor = all_descriptor[:int(len(all_descriptor) / 10)]
    print('{} 共计特征点描述:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), len(all_descriptor)))
    if os.path.exists('./model_save/kmeans-{}.m'.format(centers)):
        kmeans = joblib.load('./model_save/kmeans-{}.m'.format(centers))
        print('loaded {}'.format('./model_save/kmeans-{}.m'.format(centers)))
        param = {'max_iter': 2}
        kmeans = kmeans.set_params(**param)
    else:
        kmeans = KMeans(n_clusters=centers
                        , precompute_distances=True  # 总是预先计算距离
                        , max_iter=1  # 最大迭代次数
                        , n_jobs=-1  # 使用所有CPU计算资源
                        )
    # 尝试使用 MiniBatchKMeans 应该可以显著的减少计算量
    # if os.path.exists('./model_save/minibatch-kmeans-{}.m'.format(centers)):
    #     kmeans = joblib.load('./model_save/minibatch-kmeans-{}.m'.format(centers))
    #     print('loaded {}'.format('./model_save/minibatch-kmeans-{}.m'.format(centers)))
    #     param = {'max_iter': 7000,'max_no_improvement': 100}
    #     kmeans = kmeans.set_params(**param)
    # else:
    #     kmeans = MiniBatchKMeans(n_clusters=centers
    #                              , batch_size=int(len(all_descriptor) / 100)
    #                              , max_iter=7000  # 最大迭代次数
    #                              , max_no_improvement=100
    #                              )
    kmeans.fit(all_descriptor)
    print(kmeans.get_params())
    # 要获得预测向量属于哪个类别/要获得与类别中心的距离
    print('{} k-means完毕'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
    # 保存模型
    joblib.dump(kmeans, './model_save/kmeans-{}.m'.format(centers))

    # 加载模型
    # kmeans = joblib.load('./model_save/kmeans-{}.m'.format(centers))
    # 500 个中心
    # print(kmeans.cluster_centers_)
    # 每个样本所属的簇
    # print(kmeans.labels_)
    # 用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
    # print(kmeans.inertia_)

    # 计算簇质心并给每个样本预测类别。
    # test1 = kmeans.fit_predict(x)
    # 计算簇并 transform X to cluster-distance space。
    # test2 = kmeans.fit_transform(x)

    # x = np.reshape(all_descriptor[0], (1, -1))
    # 给每个样本估计最接近的簇。
    # test3 = kmeans.predict(x)

    # Opposite of the value of X on the K-means objective.
    # test4 = kmeans.score(x)  # 等价于距离计算
    # distance = np.power(x.reshape((-1, 1)) - kmeans.cluster_centers_[test3].reshape((-1, 1)), 2).sum()


def get_surf_descriptor(filenames):
    surf = SURF()
    result = []
    for f in filenames:
        img, keypoint, d = surf.get_surf(f)
        if d is not None:
            result.extend(d)
    return result


if __name__ == '__main__':
    paths = ['/home/liutao/homework/data/oxbuild_images/']
    build_sample_center(paths)
