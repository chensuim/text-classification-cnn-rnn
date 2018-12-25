# coding: utf-8

from __future__ import print_function

import os
import tensorflow as tf
import tensorflow.contrib.keras as kr

from cnn_model import TCNNConfig, TextCNN
from data.cnews_loader import read_category, read_vocab

try:
    bool(type(unicode))
except NameError:
    unicode = str

base_dir = 'data/listen'
vocab_dir = os.path.join(base_dir, 'vocab.txt')

save_dir = 'checkpoints/textcnn'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


class CnnModel:
    def __init__(self):
        self.config = TCNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.model = TextCNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=save_path)  # 读取保存的模型

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        data = [self.word_to_id[x] for x in content.split() if x in self.word_to_id]

        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        res = []
        for i, pred_cls in enumerate(y_pred_cls[0]):
            if pred_cls > 0.5:
                res.append(self.categories[i])
        return res
        # return self.categories[y_pred_cls[0]]


if __name__ == '__main__':
    cnn_model = CnnModel()
    tp = 0.0
    fp = 0.0
    fn = 0.0
    with open('./data/listen/test_listen_fast', 'r') as f:
        i = 0
        for line in f.readlines():
            i += 1
            if i & 0xff == 0:
                print('tp:%s, fn: %s, fp: %s' % (tp, fn, fp))
                recall = tp / (tp + fn)
                precision = tp / (tp + fp)
                f1 = 2 * recall * precision / (recall + precision + 0.000001)
                print('recall: %s, precision: %s, f1: %s' % (recall, precision, f1))
            labels = list()
            text = list()
            for tag in line.split():
                if tag.startswith('__label__'):
                    labels.append(tag)
                else:
                    text.append(tag)
            text = ' '.join(text)
            res = set(cnn_model.predict(text))
       	    labels = set(labels)
            inter = res & labels
            print(inter)
            print(line)
            tp += len(inter)
            fn += len(labels) - len(inter)
            fp += len(res) - len(inter)
            break
                    
    test_demo = ['can you play the piano yes i can no he cant yes they are',
                 '1alice has a beautiful guitar 2mr smith is a great musician 3jack can do chinese kung fu very well 4do you want to join the sports club 5peter often writes to his friends on the weekend 30421']
    for i in test_demo:
        print(cnn_model.predict(i))
