#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    @version: V-17-4-13
    @author: Linlifang
    @file: lstm_api.py
    @time: 17-4-13.下午3:01
"""
from datetime import datetime
from api.core import helper
import tensorflow as tf
from api.lstm.lstm_crf import LstmCrf
from config.config import *


def train():
    """
    函数说明: 训练模型
    :return:
    """
    start = datetime.now()
    print('开始训练模型:', start, '\n')
    train_data = helper.get_train(train_path=train_path, seq_max_len=num_steps)
    num_chars, num_labels = train_data['number']

    embedding_matrix = helper.get_embedding(emb_path) if emb_path else None

    config = tf.ConfigProto(allow_soft_placement=True)
    kwarg = {'num_chars': num_chars, 'num_classes': num_labels, 'num_steps': num_steps, 'num_epochs': num_epochs,
             'embedding_matrix': embedding_matrix, 'is_training': True}

    with tf.Session(config=config) as sess:
        with tf.device(cpu_config):
            initializer = tf.random_uniform_initializer(-0.1, 0.1)
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                model = LstmCrf(**kwarg)
                tf.global_variables_initializer().run()

                model.train(sess, model_path, train_data)

                print('正确率: ', model.max_f1)
                end = datetime.now()
                print('\n结束模型训练:', end, '\n训练模型耗时:', end - start)


def test():
    """
    函数说明: 序列标注
    :return:
    """
    start = datetime.now()
    print('开始测试数据:', start, '\n')
    test_data = helper.get_test(test_path=test_path, seq_max_len=num_steps)
    num_chars, num_labels = test_data['number']

    embedding_matrix = helper.get_embedding(emb_path) if emb_path else None

    config = tf.ConfigProto(allow_soft_placement=True)
    kwarg = {'num_chars': num_chars, 'num_classes': num_labels, 'num_steps': num_steps, 'num_epochs': num_epochs,
             'embedding_matrix': embedding_matrix, 'is_training': False}

    with tf.Session(config=config) as sess:
        with tf.device(cpu_config):
            initializer = tf.random_uniform_initializer(-0.1, 0.1)
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                model = LstmCrf(**kwarg)
                saver = tf.train.Saver()
                saver.restore(sess, model_path)
                print('\n正在测试数据...')
                model.test(sess, test_data, output_path)

                end = datetime.now()
                print('\n结束测试数据:', end, '\n测试数据耗时:', end - start)
