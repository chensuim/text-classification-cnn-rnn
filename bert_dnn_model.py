# coding: utf-8

import tensorflow as tf
import numpy as np


class BetDNNConfig(object):
    """CNN配置参数"""

    num_classes = 2640  # 类别数
    seq_length = 300
    sent_vec_len = 768

    hidden_dim = 4096  # 全连接层神经元
    hidden_dim_2 = 4096  # 全连接层神经元
    hidden_dim_3 = 2048  # 全连接层神经元
    num_rnn_layers = 2

    dropout_keep_prob = 1  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class TextBertDNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.float32, [None, self.config.sent_vec_len], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.dnn()

    def dnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(self.input_x, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)
            fc2 = tf.layers.dense(fc, self.config.hidden_dim_2, name='fc2')
            fc2 = tf.contrib.layers.dropout(fc2, self.keep_prob)
            fc2 = tf.nn.relu(fc2)
            fc3 = tf.layers.dense(fc2, self.config.hidden_dim_3, name='fc3')
            fc3 = tf.contrib.layers.dropout(fc3, self.keep_prob)
            fc3 = tf.nn.relu(fc3)

            # 分类器
            self.logits = tf.layers.dense(fc3, self.config.num_classes, name='fc4')
            self.y_pred_cls = tf.nn.sigmoid(self.logits)
            # self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            cross_entropy = tf.losses.sigmoid_cross_entropy(logits=self.logits, multi_class_labels=self.input_y, weights=10)
            #self.loss = tf.reduce_mean(cross_entropy)
            self.loss = (cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("f1"):
            # 召回率
            y_pred = tf.round(self.y_pred_cls)
            correct_pred = tf.reduce_sum(tf.reduce_sum(tf.multiply(y_pred, self.input_y), axis=0), axis=0)
            should_pred = tf.reduce_sum(tf.reduce_sum(self.input_y, axis=0), axis=0)
            predicted = tf.reduce_sum(tf.reduce_sum(y_pred, axis=0), axis=0)
            theta = tf.constant(0.0001, tf.float32)
            self.recall = correct_pred / (should_pred + theta)
            self.precision = correct_pred / (predicted + theta)
            self.f1 = 2 * self.recall * self.precision / (self.recall + self.precision + theta)
'''
        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
'''
