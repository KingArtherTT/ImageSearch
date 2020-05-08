import os
import cv2
from common import ImageFeature
from Features import SURF
import joblib
import time
import pickle
from flask import Flask, render_template, request, jsonify
from common import CosSimilarity,Euclidean_Distance
import numpy as np
import random

app = Flask(__name__)
all_img_features = None
import json


def search_top_k(k, img_name, similarity=CosSimilarity()):
    anchor = None
    global all_img_features
    for f in all_img_features:
        if f.img_name == img_name:
            anchor = f
            break
    if anchor is None:
        raise ValueError('错误的图片名称：%s' % img_name)
    index_sim = []
    for i in range(len(all_img_features)):
        index_sim.append([i, similarity.get_similarity(anchor.feature, all_img_features[i].feature)])
    # 排序 找出top-k
    index_sim = np.array(index_sim)
    # print(index_sim.shape)
    # print(index_sim[:,1].argsort()[::-1])
    sorted_index = index_sim[:,1].argsort()[::-1][1:k + 1]  # 排除自身
    print(sorted_index)
    result = []
    for i in sorted_index:
        result.append(all_img_features[i].img_name)
    return result


@app.route('/')
def Demo():
    return render_template('homepage.html')


@app.route('/action/get_result/', methods=['POST'])
def get_search_result():
    # img_name = request.form['img_name']
    img_name = request.get_json()
    # print(img_name)
    img_name = img_name['img_name']
    # print(img_name)
    global all_img_features
    if all_img_features is None:
        if os.path.exists('./pickle/all_img_features.pkl'):
            with open('./pickle/all_img_features.pkl', 'rb') as f:
                all_img_features = pickle.load(f)
    result = {}
    if all_img_features is None:
        result['code'] = -1
        result['message'] = '尚未建立特征库，请先运行BuildImageFeature.py文件'
    else:
        result['code'] = 1
        result['img_names'] = search_top_k(4, img_name)
    return jsonify(result)


@app.route('/action/change_search/',methods=['POST'])
def change_search():
    path = './static/img_data/oxbuild_images/'
    all_filenames = os.listdir(path)
    random.shuffle(all_filenames)
    return jsonify(all_filenames[:12])
    # 1.建立Flask 服务
    # 2.编写接口  接收 上传图片
    # 3.返回Top5的图片路径
