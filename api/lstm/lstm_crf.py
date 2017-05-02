#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    @version: V-17-4-13
    @author: Linlifang
    @file: lstm_crf.py
    @time: 17-4-13.下午2:55
"""
import math
from api.core import helper
import numpy as np
import tensorflow as tf


class LstmCrf(object):
    """
    LSTM + CRF 分词
    """

    def __init__(self, num_chars, num_classes, num_steps=200, num_epochs=100, embedding_matrix=None, is_training=True,
                 is_crf=True, weight=False):
        """
        函数说明: 类初始化
        :param num_chars: 字符个数
        :param num_classes: 标签个数
        :param num_steps: 句子步长, 默认200
        :param num_epochs: 迭代次数, 默认100
        :param embedding_matrix: 词向量
        :param is_training: 是否训练
        :param is_crf: crf分词
        :param weight:
        """
        # 参数
        self.max_f1 = 0
        self.learning_rate = 0.002
        self.dropout_rate = 0.5
        self.batch_size = 128
        self.num_layers = 1
        self.emb_dim = 100
        self.hidden_dim = 100
        self.num_epochs = num_epochs
        self.num_steps = num_steps
        self.num_chars = num_chars
        self.num_classes = num_classes

        # 占位符
        self.inputs = tf.placeholder(tf.int32, [None, self.num_steps])  # 句子长度
        self.targets = tf.placeholder(tf.int32, [None, self.num_steps])  # 标签长度
        self.targets_weight = tf.placeholder(tf.float32, [None, self.num_steps])  # 权值
        self.targets_transition = tf.placeholder(tf.int32, [None])

        # 词嵌入
        if embedding_matrix:
            self.embedding = tf.Variable(embedding_matrix, trainable=False, name="emb", dtype=tf.float32)
        else:
            self.embedding = tf.get_variable("emb", [self.num_chars, self.emb_dim])
        self.inputs_emb = tf.nn.embedding_lookup(self.embedding, self.inputs)
        self.inputs_emb = tf.transpose(self.inputs_emb, [1, 0, 2])
        self.inputs_emb = tf.reshape(self.inputs_emb, [-1, self.emb_dim])
        self.inputs_emb = tf.split(0, self.num_steps, self.inputs_emb)

        # lstm 神经元,隐藏层100
        lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
        lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)

        # dropout 避免过拟合
        if is_training:
            lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_fw, output_keep_prob=(1 - self.dropout_rate))
            lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_bw, output_keep_prob=(1 - self.dropout_rate))

        # 层数
        lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_fw] * self.num_layers)
        lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_bw] * self.num_layers)

        # get the length of each sample
        self.length = tf.reduce_sum(tf.sign(self.inputs), reduction_indices=1)
        self.length = tf.cast(self.length, tf.int32)

        # forward and backward
        self.outputs, _, _ = tf.nn.bidirectional_rnn(lstm_cell_fw, lstm_cell_bw, self.inputs_emb, dtype=tf.float32,
                                                     sequence_length=self.length)

        # softmax
        self.outputs = tf.reshape(tf.concat(1, self.outputs), [-1, self.hidden_dim * 2])
        self.softmax_w = tf.get_variable("softmax_w", [self.hidden_dim * 2, self.num_classes])
        self.softmax_b = tf.get_variable("softmax_b", [self.num_classes])
        self.logits = tf.matmul(self.outputs, self.softmax_w) + self.softmax_b

        if not is_crf:
            pass
        else:
            self.tags_scores = tf.reshape(self.logits, [self.batch_size, self.num_steps, self.num_classes])
            self.transitions = tf.get_variable("transitions", [self.num_classes + 1, self.num_classes + 1])

            dummy_val = -1000
            class_pad = tf.Variable(dummy_val * np.ones((self.batch_size, self.num_steps, 1)), dtype=tf.float32)
            self.observations = tf.concat(2, [self.tags_scores, class_pad])

            begin_vec = tf.Variable(np.array([[dummy_val] * self.num_classes + [0] for _ in range(self.batch_size)]),
                                    trainable=False, dtype=tf.float32)
            end_vec = tf.Variable(np.array([[0] + [dummy_val] * self.num_classes for _ in range(self.batch_size)]),
                                  trainable=False, dtype=tf.float32)
            begin_vec = tf.reshape(begin_vec, [self.batch_size, 1, self.num_classes + 1])
            end_vec = tf.reshape(end_vec, [self.batch_size, 1, self.num_classes + 1])

            self.observations = tf.concat(1, [begin_vec, self.observations, end_vec])

            self.mask = tf.cast(tf.reshape(tf.sign(self.targets), [self.batch_size * self.num_steps]), tf.float32)

            # point score
            self.point_score = tf.gather(tf.reshape(self.tags_scores, [-1]),
                                         tf.range(0, self.batch_size * self.num_steps) * self.num_classes + tf.reshape(
                                             self.targets, [self.batch_size * self.num_steps]))
            self.point_score *= self.mask

            # transition score
            self.trans_score = tf.gather(tf.reshape(self.transitions, [-1]), self.targets_transition)

            # real score
            self.target_path_score = tf.reduce_sum(self.point_score) + tf.reduce_sum(self.trans_score)

            # all path score
            self.total_path_score, self.max_scores, self.max_scores_pre = self.forward(self.observations,
                                                                                       self.transitions, self.length)

            # loss
            self.loss = - (self.target_path_score - self.total_path_score)

        # summary
        self.train_summary = tf.summary.scalar("loss", self.loss)
        self.val_summary = tf.summary.scalar("loss", self.loss)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    @staticmethod
    def log_sum_exp(x, axis=None):
        """
        函数说明: log
        :param x:
        :param axis:
        :return:
        """
        x_max = tf.reduce_max(x, reduction_indices=axis, keep_dims=True)
        x_max_ = tf.reduce_max(x, reduction_indices=axis)
        return x_max_ + tf.log(tf.reduce_sum(tf.exp(x - x_max), reduction_indices=axis))

    def forward(self, observations, transitions, length, is_viterbi=True, return_best_seq=True):
        length = tf.reshape(length, [self.batch_size])
        transitions = tf.reshape(tf.concat(0, [transitions] * self.batch_size), [self.batch_size, 6, 6])
        observations = tf.reshape(observations, [self.batch_size, self.num_steps + 2, 6, 1])
        observations = tf.transpose(observations, [1, 0, 2, 3])
        previous = observations[0, :, :, :]
        max_scores = []
        max_scores_pre = []
        alphas = [previous]
        for t in range(1, self.num_steps + 2):
            previous = tf.reshape(previous, [self.batch_size, 6, 1])
            current = tf.reshape(observations[t, :, :, :], [self.batch_size, 1, 6])
            alpha_t = previous + current + transitions
            if is_viterbi:
                max_scores.append(tf.reduce_max(alpha_t, reduction_indices=1))
                max_scores_pre.append(tf.argmax(alpha_t, dimension=1))
            alpha_t = tf.reshape(self.log_sum_exp(alpha_t, axis=1), [self.batch_size, 6, 1])
            alphas.append(alpha_t)
            previous = alpha_t

        alphas = tf.reshape(tf.concat(0, alphas), [self.num_steps + 2, self.batch_size, 6, 1])
        alphas = tf.transpose(alphas, [1, 0, 2, 3])
        alphas = tf.reshape(alphas, [self.batch_size * (self.num_steps + 2), 6, 1])

        last_alphas = tf.gather(alphas, tf.range(0, self.batch_size) * (self.num_steps + 2) + length)
        last_alphas = tf.reshape(last_alphas, [self.batch_size, 6, 1])

        max_scores = tf.reshape(tf.concat(0, max_scores), (self.num_steps + 1, self.batch_size, 6))
        max_scores_pre = tf.reshape(tf.concat(0, max_scores_pre), (self.num_steps + 1, self.batch_size, 6))
        max_scores = tf.transpose(max_scores, [1, 0, 2])
        max_scores_pre = tf.transpose(max_scores_pre, [1, 0, 2])

        return tf.reduce_sum(self.log_sum_exp(last_alphas, axis=1)), max_scores, max_scores_pre

    def train(self, sess, save_file, train_data):
        """
        函数说明: 训练数据
        :param sess:
        :param save_file: 模型文件路径
        :param train_data:
        :return:
        """
        saver = tf.train.Saver()
        x_train, y_train, x_val, y_val = train_data['train']
        char_id, id_char, label_id, id_label = train_data['token']

        merged = tf.summary.merge_all()
        summary_writer_train = tf.summary.FileWriter('loss_log/train_loss', sess.graph)
        summary_writer_val = tf.summary.FileWriter('loss_log/val_loss', sess.graph)

        # 每次训练128条语句
        num_iterations = int(math.ceil(1.0 * len(x_train) / self.batch_size))

        cnt = 0
        for epoch in range(self.num_epochs):
            # shuffle train in each epoch
            sh_index = np.arange(len(x_train))
            np.random.shuffle(sh_index)
            x_train = x_train[sh_index]
            y_train = y_train[sh_index]
            print("\n当前迭代次数: ", epoch)
            for iteration in range(num_iterations):
                # train 选取128条数据
                x_train_batch, y_train_batch = helper.next_batch(x_train, y_train,
                                                                 start_index=iteration * self.batch_size,
                                                                 batch_size=self.batch_size)
                y_train_weight_batch = 1 + np.array((y_train_batch == label_id['B']) | (y_train_batch == label_id['E']),
                                                    float)
                transition_batch = helper.get_transition(y_train_batch)

                data = [self.optimizer, self.loss, self.max_scores, self.max_scores_pre, self.length,
                        self.train_summary]
                feed_dict = {self.targets_transition: transition_batch, self.inputs: x_train_batch,
                             self.targets: y_train_batch, self.targets_weight: y_train_weight_batch}

                _, loss_train, max_scores, max_scores_pre, length, train_summary = sess.run(data, feed_dict=feed_dict)

                predicts_train = self.viterbi(max_scores, max_scores_pre, length, predict_size=self.batch_size)

                # 训练集
                if iteration % 10 == 0:
                    cnt += 1
                    precision_train, recall_train, f1_train = self.evaluate(x_train_batch, y_train_batch,
                                                                            predicts_train, id_char, id_label)
                    summary_writer_train.add_summary(train_summary, cnt)
                    print("训练集::\t循环: %3d, loss: %3d, 准确率: %.3f, 召回率: %.3f, f1: %.3f"
                          % (iteration, loss_train, precision_train, recall_train, f1_train))

                # 验证集
                if iteration % 10 == 0:
                    x_val_batch, y_val_batch = helper.next_random_batch(x_val, y_val, batch_size=self.batch_size)
                    y_val_weight_batch = 1 + np.array((y_val_batch == label_id['B']) | (y_val_batch == label_id['E']),
                                                      float)
                    transition_batch = helper.get_transition(y_val_batch)

                    data = [self.loss, self.max_scores, self.max_scores_pre, self.length, self.val_summary]
                    feed_dict = {self.targets_transition: transition_batch, self.inputs: x_val_batch,
                                 self.targets: y_val_batch, self.targets_weight: y_val_weight_batch}
                    loss_val, max_scores, max_scores_pre, length, val_summary = sess.run(data, feed_dict=feed_dict)

                    predicts_val = self.viterbi(max_scores, max_scores_pre, length, predict_size=self.batch_size)
                    precision_val, recall_val, f1_val = self.evaluate(x_val_batch, y_val_batch, predicts_val, id_char,
                                                                      id_label)
                    summary_writer_val.add_summary(val_summary, cnt)
                    print("验证集::\t循环: %3d, loss: %3d, 准确率: %.3f, 召回率: %.3f, f1: %.3f"
                          % (iteration, loss_val, precision_val, recall_val, f1_val))

                    if f1_val >= self.max_f1:
                        print("\n---------------\n*保存模型.....")
                        self.max_f1 = f1_val
                        saver.save(sess, save_file)
                        print("*f1: %.4f\n---------------\n" % self.max_f1)

    def test(self, sess, test_data, output_path):
        """
        函数说明: 测试
        :param sess:
        :param test_data:
        :param output_path:
        :return:
        """
        x_test, x_test_str = test_data['test']
        char_id, id_char, label_id, id_label = test_data['token']

        num_iterations = int(math.ceil(1.0 * len(x_test) / self.batch_size))
        print("总迭代步数: ", num_iterations)

        with open(output_path, "w") as fp:
            for iterations in range(num_iterations):
                print("迭代步数: ", iterations + 1)
                test_batch, test_str_batch = helper.next_test_batch(x_test, x_test_str, iterations * self.batch_size)
                results = self.predict_batch(sess, test_batch, test_str_batch, id_label)
                for result in results:
                    fp.writelines(result + "\n")

    @staticmethod
    def viterbi(max_scores, max_scores_pre, length, predict_size=128):
        """
        函数说明: 维特比算法
        :param max_scores:
        :param max_scores_pre:
        :param length:
        :param predict_size:
        :return:
        """
        best_paths = []
        for m in range(predict_size):
            path = []
            last_max_node = np.argmax(max_scores[m][length[m]])
            # last_max_node = 0
            for t in range(1, length[m] + 1)[::-1]:
                last_max_node = max_scores_pre[m][t][last_max_node]
                path.append(last_max_node)
            path = path[::-1]
            best_paths.append(path)
        return best_paths

    def predict_batch(self, sess, x_id, x_str, id_label):
        """
        函数说明: 获取测试标签及字符
        :param sess:
        :param x_id: 测试数据id
        :param x_str: 测试数据
        :param id_label: id和字符对应关系
        :return:
        """
        length, max_scores, max_scores_pre = sess.run([self.length, self.max_scores, self.max_scores_pre],
                                                      feed_dict={self.inputs: x_id})
        predicts = self.viterbi(max_scores, max_scores_pre, length, self.batch_size)
        results = []
        for i in range(len(predicts)):
            x = ''.join(x_str[i])
            if x == 'x' * self.num_steps:
                continue
            y_pre = ''.join([id_label[str(val)] for val in predicts[i] if val != 5 and val != 0])
            entity = '_'.join(helper.extract_entity(x, y_pre))
            results.append('<@>'.join([x, entity]))
        return results

    @staticmethod
    def evaluate(x_true, y_true, y_pred, id_char, id_label):
        """
        函数说明: 计算识别率
        :param x_true: 字符id
        :param y_true: 标签id
        :param y_pred: 预测标签id
        :param id_char:
        :param id_label:
        :return: precision, recall, f1 准确率, 召回率, 精度
        """
        precision = -1.0
        recall = -1.0
        f1 = -1.0
        hit_num = 0
        pred_num = 0
        true_num = 0
        for i in range(len(y_true)):
            x = ''.join([str(id_char[val]) for val in x_true[i]])
            y = ''.join([str(id_label[val]) for val in y_true[i]])
            y_hat = ''.join([id_label[val] for val in y_pred[i] if val != 5])

            true_labels = helper.extract_entity(x, y)
            pred_labels = helper.extract_entity(x, y_hat)

            hit_num += len(set(true_labels) & set(pred_labels))
            pred_num += len(set(pred_labels))
            true_num += len(set(true_labels))
        if pred_num != 0:
            precision = 1.0 * hit_num / pred_num
        if true_num != 0:
            recall = 1.0 * hit_num / true_num
        if precision > 0 and recall > 0:
            f1 = 2.0 * (precision * recall) / (precision + recall)
        return precision, recall, f1
