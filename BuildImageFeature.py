from common import ImageFeature
from Features import SURF
import joblib
import os
import pickle
import time


def main():
    # global all_img_features
    # 加载k-means模型  建立所有图片的特征库-ImageFeature
    path = './static/img_data/oxbuild_images/'
    all_filenames = os.listdir(path)

    surf = SURF()
    kmeans_model = joblib.load('./model_save/minibatch-kmeans-500.m')
    surf.set_kmeans_model(kmeans_model)
    if os.path.exists('./pickle/all_img_features.pkl'):
        with open('./pickle/all_img_features.pkl', 'rb') as f:
            all_img_features = pickle.load(f)
    else:
        all_img_features = []
        i = 0
        for f in all_filenames:
            feature = surf.get_feature(path + f)
            all_img_features.append(ImageFeature(f, feature))
            i += 1
            if i % 100 == 0:
                print('{} 进度：{}/{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), i, len(all_filenames)))

        with open('./pickle/all_img_features.pkl', 'wb') as f:
            pickle.dump(all_img_features, f)
    print('{} 建立特征库完毕，共计:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                                    len(all_img_features)))


if __name__ == '__main__':
    main()
