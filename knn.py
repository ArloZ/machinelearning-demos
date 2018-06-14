# -*- coding: utf-8 -*-
"""
K-近邻算法
"""
import numpy as np

from datasets import dataset
import utils


def classify(test_data, data, label, k=6):
    data_set_size = data.shape[0]

    # 输入测试样例对数据集中的每个样本进行特征距离计算
    diff = np.tile(test_data, (data_set_size, 1)) - data
    sq_diff = diff ** 2
    sq_sum = sq_diff.sum(1)
    distance = sq_sum ** 0.5

    # 输入测试样例与数据集中每个样本的距离排序
    sorted_indicies = distance.argsort()

    # 统计距离最近的K个样本
    label_count = {}
    for i in range(k):
        vote_label = label[sorted_indicies[i]]
        label_count[vote_label] = label_count.get(vote_label, 0) + 1

    # 样本标签最多的label作为返回结果
    sorted_label = sorted(label_count.items(), key=lambda item: item[1], reverse=True)
    return sorted_label[0][0]


def test_dating():
    test_ratio = 0.1
    data, label = dataset.dating_set(norm=True)
    utils.plot_scatter(data[:, 2], data[:, 1], 10 * label, 15 * label)
    test_size = int(data.shape[0] * test_ratio)
    right_count = 0
    for i in range(test_size):
        predict_label = classify(data[i, :], data[test_size:, :], label[test_size:])
        right_count += 1 if label[i] == predict_label else 0
    print("accuracy: %f" % float(right_count / test_size))


def handwriting():
    pass


if __name__ == '__main__':
    test_dating()
