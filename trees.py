# -*- coding: utf-8 -*-
"""
决策树算法
"""
import math

import numpy as np
from datasets import dataset


def calc_entropy(datas):
    """
    计算信息熵
    :return:
    """
    nums = len(datas)
    label_count = {}
    for feat_vec in datas:
        label = feat_vec[-1]
        label_count[label] = label_count.get(label, 0) + 1
    entropy = 0.0
    for label in label_count:
        prob = float(label_count[label]) / nums
        entropy -= prob * math.log(prob, 2)
    # print(label_count, entropy)
    return entropy


def split_data_set(datas, axis, value):
    return_data = []
    for feat_vec in datas:
        if feat_vec[axis] == value:
            obtain_feat = feat_vec[:axis]
            obtain_feat.extend(feat_vec[axis + 1:])
            return_data.append(obtain_feat)
    return return_data


def choose_best_feature(datas):
    base_entropy = calc_entropy(datas)
    feat_len = len(datas[0]) - 1
    best_feat = -1
    best_gain = 0.0
    for i in range(feat_len):
        feat_values = [example[i] for example in datas]
        uniq_feat_values = set(feat_values)
        new_entropy = 0.0
        for feat_value in uniq_feat_values:
            split_data = split_data_set(datas, i, feat_value)
            prob = float(len(split_data)) / float(len(datas))
            new_entropy += prob * calc_entropy(split_data)
        info_gain = base_entropy - new_entropy
        if info_gain > best_gain:
            best_gain = info_gain
            best_feat = i
    return best_feat


def major_cnt(label_list):
    labels = {}
    for label in label_list:
        labels[label] = labels.get(label, 0) + 1
    labels = sorted(labels.items(), key=lambda item: item[1], reverse=True)
    return labels[0][0]


def create_tree(datas, feat_label):
    # 类别完全相同则直接返回
    label_list = [example[-1] for example in datas]
    if label_list.count(label_list[0]) == len(label_list):
        return label_list[0]
    # 已遍历完所有特征，则取类别最多的作为结果返回
    if len(datas[0]) == 1:
        return major_cnt(label_list)
    best_feat = choose_best_feature(datas)
    best_feat_label = feat_label[best_feat]
    del (feat_label[best_feat])
    tree = {best_feat_label: {}}
    feats = [example[best_feat] for example in datas]
    uniq_feats = set(feats)
    for value in uniq_feats:
        sub_label = feat_label[:]
        tree[best_feat_label][value] = create_tree(split_data_set(datas, best_feat, value), sub_label)

    return tree


def classify(tree, feat_label, test_vec):
    first_feat = list(tree.keys())[0]
    feat_idx = feat_label.index(first_feat)
    sub_tree = tree[first_feat]
    for key in sub_tree.keys():
        if test_vec[feat_idx] == key:
            if type(sub_tree[key]).__name__ == 'dict':
                return classify(sub_tree[key], feat_label, test_vec)
            else:
                return sub_tree[key]


if __name__ == '__main__':
    data, feat_label = dataset.fish_set()
    tree = create_tree(data, feat_label[:])
    print(tree)
    test_vec = [1, 1]
    predict = classify(tree, feat_label, test_vec)
    print('test_vec:', test_vec, 'predict:', predict)
