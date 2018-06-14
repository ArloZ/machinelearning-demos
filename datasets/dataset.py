# -*- coding: utf-8 -*-
import os
import numpy as np


def base_dir():
    current_file = os.path.abspath(__file__)
    base_file = os.path.basename(current_file)
    return current_file.split(base_file)[0]


def dating_set(norm=False):
    f = os.path.join(base_dir(), "datingTestSet.txt")
    print("load datasets from: %s" % f)
    with open(f, encoding='utf-8') as f:
        lines = f.readlines()
        _data = np.zeros((len(lines), 3))
        _label = []
        index = 0
        for line in lines:
            line_parts = line.strip().split('\t')
            _data[index, :] = line_parts[0:3]
            _label.append(int(line_parts[-1]))
            index += 1
        print("data shape:", _data.shape, "label size: %s" % len(_label))
        if not norm:
            return _data, np.array(_label)
        # normalize data set
        min_values = _data.min(0)
        max_values = _data.max(0)
        _norm_data = (_data - min_values) / (max_values - min_values)
        return _norm_data, np.array(_label)


def fish_set():
    _data_set = [[1, 1, 'Y'], [1, 1, 'Y'], [0, 1, 'N'], [1, 0, 'N'], [0, 1, 'N'], [0, 0, 'N'], [0, 1, 'M']]
    return _data_set, ['f1', 'f2']


if __name__ == '__main__':
    data, label = dating_set(norm=True)
    print(data, label)
