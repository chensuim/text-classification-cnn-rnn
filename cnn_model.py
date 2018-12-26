# coding: utf-8

import tensorflow as tf
import numpy as np


class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 300  # 词向量维度
    seq_length = 300  # 序列长度
    num_classes = 2640  # 类别数
    num_filters = 128 * 16 # 卷积核数目
    kernel_sizes = [2,4,7]  # 卷积核尺寸
    vocab_size = 5000  # 词汇表达小

    hidden_dim = 4096  # 全连接层神经元
    hidden_dim_2 = 2048  # 全连接层神经元
    hidden_dim_3 = 2048  # 全连接层神经元
    num_rnn_layers = 2
    l2_reg_lambda = 0.0

    dropout_keep_prob = 1  # dropout保留比例
    start_learning_rate = 1e-5  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 100000  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config, word2vec, word2id):
        self.config = config
        embedding_matrix = np.zeros((self.config.vocab_size, self.config.embedding_dim))
        for word in word2vec:
            embedding_matrix[word2id[word]] = word2vec[word]
            self.embedding_matrix = embedding_matrix

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def cnn(self):
        """CNN模型"""
        def lstm_cell():   # lstm核
            return tf.contrib.rnn.BasicLSTMCell(self.config.hidden_dim, state_is_tuple=True)

        def dropout():
            cell = lstm_cell()
            return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim],
                                        initializer=tf.constant_initializer(self.embedding_matrix), trainable=False
                                        )
            #embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)
        pooled_outputs = []

        #for filter_size in self.config.kernel_sizes:
        #    with tf.name_scope("conv-filter{0}".format(filter_size)):
        #        conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, filter_size, name='conv_%s' % filter_size, padding="same")
	#	pooled = tf.layers.max_pooling1d(inputs=conv, pool_size=self.config.seq_length, strides=self.config.seq_length)
	#	print('pooled size is %s' % pooled.shape)
        #        # pooled = tf.reduce_max(conv, reduction_indices=[1], name='pooled_%s' % filter_size)

        #    pooled_outputs.append(pooled)

        ## Combine all the pooled features
        #num_filters_total = self.config.num_filters * len(self.config.kernel_sizes)
        #pool = tf.concat(pooled_outputs, axis=1)
	#print('concat pool shape is %s' % pool.shape)
        #pool_flat = tf.reshape(pool, shape=[-1, num_filters_total])
        #print('pool flat shape is %s' % pool_flat.shape)
        conv1 = tf.layers.conv1d(inputs=embedding_inputs, filters=self.config.num_filters, kernel_size=5, padding='same', activation=tf.nn.relu, kernel_initializer=tf.initializers.truncated_normal(stddev=0.3)) 
	pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=self.config.seq_length, strides=self.config.seq_length)
	pool1 = tf.reshape(pool1, shape=[-1, self.config.num_filters])
	#conv2 = tf.layers.conv1d(inputs=pool1, filters=self.config.num_filters*2, kernel_size=5, padding='same', activation=tf.nn.relu)
	#pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=self.config.seq_length/2, strides=self.config.seq_length/2)
	#pool_flat = tf.reshape(pool2, shape=[-1, self.config.num_filters * 2])


       # with tf.name_scope("cnn"):
       #     # CNN layer
       #     conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
       #     # global max pooling layer
       #     gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(pool1, self.config.hidden_dim, name='fc1', kernel_initializer=tf.initializers.truncated_normal(stddev=0.3))
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)
            #fc2 = tf.layers.dense(fc, self.config.hidden_dim_2, name='fc2')
            #fc2 = tf.contrib.layers.dropout(fc2, self.keep_prob)
            #fc2 = tf.nn.relu(fc2)
            #fc3 = tf.layers.dense(fc2, self.config.hidden_dim_3, name='fc3')
            #fc3 = tf.contrib.layers.dropout(fc3, self.keep_prob)
            #fc3 = tf.nn.relu(fc3)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc4')
            self.y_pred_cls = tf.nn.sigmoid(self.logits)
            # self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            cross_entropy = tf.nn.weighted_cross_entropy_with_logits(logits=self.logits, targets=self.input_y, pos_weight=2)
	    cross_entropy = tf.reduce_mean(cross_entropy)
            l2_losses = 0#tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()],
                        #         name="l2_losses") * self.config.l2_reg_lambda
	    self.loss = tf.add(cross_entropy, l2_losses, name="loss")
            #self.loss = (cross_entropy)
            # decayed learninig rate
	    global_step = tf.Variable(0, trainable=False)
	    learning_rate = tf.train.exponential_decay(self.config.start_learning_rate, global_step, 10000, 0.99, staircase=True)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss, global_step=global_step)

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
