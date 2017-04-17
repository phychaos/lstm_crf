#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    @version: V-17-4-13
    @author: Linlifang
    @file: lstm_api.py
    @time: 17-4-13.下午3:01
"""
import os
import re
import numpy as np
import pandas as pd


def get_embedding(file_path="embedding"):
    char_id, id_char = load_map("token/char_id")
    row_index = 0
    with open(file_path, "r") as fp:
        for row in fp.readlines():
            line = row.strip().split()
            row_index += 1
            if row_index == 1:
                num_chars = int(line[0])
                emb_dim = int(line[1])
                emb_matrix = np.zeros((len(char_id), emb_dim))
                continue
            char = line[0]
            emb_vec = [float(val) for val in line[1:]]
            if char in char_id:
                emb_matrix[char_id[char]] = emb_vec
    return emb_matrix


def extract_entity(sentence, labels):
    """
    函数说明: 获取标签对应的字符值
    :param sentence:
    :param labels:
    :return:
    """
    entitys = []
    pattern = re.compile(r'B*I*E')
    for kk in pattern.finditer(labels):
        start, end = kk.span()
        entity = sentence[start: end]
        entitys.append(entity)
    return entitys


def next_batch(x, y, start_index, batch_size=128):
    """
    函数说明: 截取句子片段
    :param x: 字符
    :param y:  标签
    :param start_index: 开始索引
    :param batch_size: 大小
    :return: x[start_index: start_index+batch_size]
    """
    last_index = start_index + batch_size
    x_batch = list(x[start_index: min(last_index, len(x))])
    y_batch = list(y[start_index: min(last_index, len(x))])
    if last_index > len(x):
        left_size = last_index - (len(x))
        for i in range(left_size):
            index = np.random.randint(len(x))
            x_batch.append(x[index])
            y_batch.append(y[index])
    x_batch = np.array(x_batch)
    y_batch = np.array(y_batch)
    return x_batch, y_batch


def next_random_batch(x, y, batch_size=128):
    """
    函数说明: 随机选取验证集
    :param x:
    :param y:
    :param batch_size:
    :return:
    """
    x_batch = []
    y_batch = []
    for i in range(batch_size):
        index = np.random.randint(len(x))
        x_batch.append(x[index])
        y_batch.append(y[index])
    x_batch = np.array(x_batch)
    y_batch = np.array(y_batch)
    return x_batch, y_batch


def next_test_batch(x, y, start_index, num_steps=200, batch_size=128):
    """
    函数说明: 截取句子片段
    :param x: 字符
    :param y:  标签
    :param start_index: 开始索引
    :param num_steps: 句子长度
    :param batch_size: 大小
    :return: x[start_index: start_index+batch_size]
    """
    last_index = start_index + batch_size
    x_batch = list(x[start_index: min(last_index, len(x))])
    y_batch = list(y[start_index: min(last_index, len(x))])
    if last_index > len(x):
        left_size = last_index - (len(x))
        for i in range(left_size):
            x_batch.append([0] * num_steps)
            y_batch.append(['x'] * num_steps)
    x_batch = np.array(x_batch)
    y_batch = np.array(y_batch)
    return x_batch, y_batch


def padding(sample, seq_max_len):
    """
    函数说明: add数组长度为seq_max_len
    :param sample:
    :param seq_max_len: 句子长度
    :return:
    """
    for i in range(len(sample)):
        if len(sample[i]) < seq_max_len:
            sample[i] += [0 for _ in range(seq_max_len - len(sample[i]))]
    return sample


def prepare(chars, labels, seq_max_len, is_padding=True):
    """
    函数说明: -1分离句子，取句子长度为200字,转化为数组
    :param chars: 字符-id
    :param labels: 标签-id
    :param seq_max_len: 句子长度,默认200
    :param is_padding: 不足200补0
    :return: 数组
    """
    x = []
    y = []
    tmp_x = []
    tmp_y = []
    for c, l in zip(chars, labels):
        if l == -1:
            if len(tmp_x) <= seq_max_len and tmp_x and tmp_y:
                x.append(tmp_x), y.append(tmp_y)
            tmp_x = []
            tmp_y = []
        else:
            tmp_x.append(c), tmp_y.append(l)
    if is_padding:
        x = np.array(padding(x, seq_max_len))
    else:
        x = np.array(x)
    y = np.array(padding(y, seq_max_len))
    return x, y


def load_map(file_path):
    """
    函数说明: 生成id-字符, 字符-id字典
    :param file_path: 文件路径
    :return: 
    """
    if not os.path.isfile(file_path):
        print("文件不存在,生成map")
        exit()

    token_id = {}
    id_token = {}
    with open(file_path) as fp:
        for row in fp.readlines():
            line = row.strip().split('\t')
            token, key_id = [i for i in line[0:2]]
            token_id[token] = key_id
            id_token[key_id] = token
    return token_id, id_token


def save_map(id_char, id_label):
    """
    函数说明: 保存数字-字符、标签
    :param id_char: id, char对应关系
    :param id_label: id, label对应关系
    :return:
    """
    with open("token/char_id", "w") as fp:
        for idx, char in id_char.items():
            fp.writelines(char + "\t" + str(idx) + "\n")
    with open("token/label_id", "w") as fp:
        for idx, label in id_label.items():
            fp.writelines(label + "\t" + str(idx) + "\n")


def build_map(train_path):
    """
    函数说明: 字符、标签数字化
    :param train_path: 训练数据路径
    :return: 字典id-char, id-label
    """
    df_train = pd.read_csv(train_path, delimiter='\t', skip_blank_lines=False, header=None, names=["char", "label"])

    # 提取字符及标签
    chars = list(set(df_train["char"][df_train["char"].notnull()]))
    labels = list(set(df_train["label"][df_train["label"].notnull()]))

    # 字符标签数字化{'O': 1, 'I':2, 'B': 3, 'E': 4}
    char_id = dict(zip(chars, range(1, len(chars) + 1)))
    label_id = dict(zip(labels, range(1, len(labels) + 1)))

    # 数字-->字符标签
    id_char = dict(zip(range(1, len(chars) + 1), chars))
    id_label = dict(zip(range(1, len(labels) + 1), labels))

    # 开始字符<PAD>
    id_char[0] = "<PAD>"
    id_label[0] = "<PAD>"
    char_id["<PAD>"] = 0
    label_id["<PAD>"] = 0

    # 新字符<NEW>
    id_char[len(chars) + 1] = "<NEW>"
    char_id["<NEW>"] = len(chars) + 1
    save_map(id_char, id_label)
    return char_id, id_char, label_id, id_label


def get_train(train_path, val_path, train_val_ratio=0.9, seq_max_len=200):
    """
    函数说明: 获取训练数据
    :param train_path: 训练数据路径
    :param val_path: 验证数据路径
    :param train_val_ratio: 训练集,验证集比例
    :param seq_max_len: 序列长度
    :return:
    """
    char_id, id_char, label_id, id_label = build_map(train_path)
    df_train = pd.read_csv(train_path, delimiter='\t', skip_blank_lines=False, header=None, names=["char", "label"])

    # 字符、标签转化为数字, 空字符(句子)转化为-1
    df_train["char_id"] = df_train.char.map(lambda xx: -1 if str(xx) == str(np.nan) else char_id[xx])
    df_train["label_id"] = df_train.label.map(lambda xx: -1 if str(xx) == str(np.nan) else label_id[xx])

    # 转换为数组n*200
    x, y = prepare(df_train["char_id"], df_train["label_id"], seq_max_len)

    # 随机排列句子顺序
    num_samples = len(x)
    index = np.arange(num_samples)
    np.random.shuffle(index)
    x = x[index]
    y = y[index]

    # 训练数据集和验证数据集
    if val_path:
        x_train = x
        y_train = y
        x_val, y_val = get_test(test_path=val_path, is_validation=True, seq_max_len=seq_max_len)['test']
    else:
        val_num = int(num_samples * train_val_ratio)
        x_train = x[:val_num]
        y_train = y[:val_num]
        x_val = x[val_num:]
        y_val = y[val_num:]

    print("训练集大小:", len(x_train), "验证集大小:", len(y_val))
    num_chars = len(id_char)
    num_labels = len(id_label)
    data = {'train': [x_train, y_train, x_val, y_val], 'token': [char_id, id_char, label_id, id_label],
            'number': [num_chars, num_labels]}
    return data


def get_test(test_path, is_validation=False, seq_max_len=200, token_path='token/'):
    """
    函数说明: 获取测试数据
    :param test_path: 测试数据路径
    :param is_validation:
    :param seq_max_len:
    :param token_path:
    :return:
    """
    char_id, id_char = load_map(token_path + "char_id")
    label_id, id_label = load_map(token_path + "label_id")

    df_test = pd.read_csv(test_path, delimiter='\t', skip_blank_lines=False, header=None, names=["char", "label"])

    def map_func(x, chars_id):
        if str(x) == str(np.nan):
            return -1
        return chars_id[x] if char_id.get(x) else chars_id["<NEW>"]

    # 字符标签转换为数字
    df_test["char_id"] = df_test.char.map(lambda x: map_func(x, char_id))
    df_test["label_id"] = df_test.label.map(lambda x: -1 if str(x) == str(np.nan) else label_id[x])

    # 数字数组化200一句,不足200补0
    x_test, y_test = prepare(df_test["char_id"], df_test["label_id"], seq_max_len)
    x_test_str, _ = prepare(df_test["char"], df_test["char_id"], seq_max_len, is_padding=False)

    print("测试数据大小", len(x_test))
    num_chars = len(id_char)
    num_labels = len(id_label)
    if is_validation:
        data = {'test': [x_test, y_test], 'token': [char_id, id_char, label_id, id_label],
                'number': [num_chars, num_labels]}
        return data
    else:
        data = {'test': [x_test, x_test_str], 'token': [char_id, id_char, label_id, id_label],
                'number': [num_chars, num_labels]}
        return data


def get_transition(y_batch):
    """
    函数说明:
    :param y_batch:
    :return:
    """
    transition_batch = []
    for m in range(len(y_batch)):
        y = [5] + list(y_batch[m]) + [0]
        for t in range(len(y)):
            if y[t] == 0:
                break
            transition_batch.append(y[t] * 6 + y[t + 1])
    transition_batch = np.array(transition_batch)
    return transition_batch
