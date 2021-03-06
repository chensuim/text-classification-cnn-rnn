# coding: utf-8
import json
import sys
from collections import Counter
import json

import numpy as np
import tensorflow.contrib.keras as kr
#from bert_serving.client import BertClient
#bc_client = BertClient(show_server_config=False)


def bert_encode(text):
    #return bc_client.encode([text])[0]
    return


if sys.version_info[0] > 2:
    is_py3 = True
else:
    reload(sys)
    sys.setdefaultencoding("utf-8")
    is_py3 = False


def native_word(word, encoding='utf-8'):
    """如果在python2下面使用python3训练的模型，可考虑调用此函数转化一下字符编码"""
    if not is_py3:
        return word.encode(encoding)
    else:
        return word


def native_content(content):
    if not is_py3:
        return content.decode('utf-8')
    else:
        return content


def open_file(filename, mode='r'):
    """
    常用文件操作，可在python2和python3间切换.
    mode: 'r' or 'w' for read or write
    """
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)


def read_file(filename):
    """读取文件数据"""
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                tags = line.strip().split()
                label = list()
                text = list()
                for tag in tags:
                    if tag.startswith('__label__'):
                        label.append(native_content(tag))
                    else:
                        text.append(native_content(tag))
                if text:
                    #contents.append(list(native_content(content)))
                    contents.append(text)
                    labels.append(label)
            except:
                pass
    return contents, labels


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    """根据训练集构建词汇表，存储"""
    data_train, _ = read_file(train_dir)

    all_data = []
    for content in data_train:
        all_data.extend(content)

    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open_file(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    words = list()
    vecs = list()
    with open_file(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        word_vecs = fp.readlines()
        for word_vec in word_vecs:
            t = word_vec.split()
            word = native_content(t[0])
            vec = np.array([float(x) for x in t[1:]])
            words.append(word)
            vecs.append(vec)
    word_to_vec = dict(zip(words, vecs))
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id, word_to_vec
    #"""读取词汇表"""
    ## words = open_file(vocab_dir).read().strip().split('\n')
    #with open_file(vocab_dir) as fp:
    #    # 如果是py2 则每个值都转化为unicode
    #    words = [native_content(_.strip()) for _ in fp.readlines()]
    #word_to_id = dict(zip(words, range(len(words))))
    #return words, word_to_id


def read_category():
    """读取分类目录，固定"""
    with open("data/ks.txt", "r") as f:
        categories = json.load(f)

    categories = [native_content(x) for x in categories]

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def to_words(content, words):
    """将id表示的内容转换为文字"""
    return ''.join(words[x] for x in content)


def to_categorical(y, num_classes=None):
    n = len(y)
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    for i in range(n):
        for cat in y[i]:
            categorical[i][cat] = 1
    return categorical


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append([cat_to_id[x] for x in labels[i]])

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad


def process_file_with_bert(filename, cat_to_id, max_length=600):
    contents, labels = read_file(filename)
    sents, label_id = [], []
    for i in range(len(contents)):
        sent = ' '.join(contents[i])
        label_id.append([cat_to_id[x] for x in labels[i]])
        sents.append(sent)
    x_pad = np.array(range(len(contents)))
    y_pad = to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示
    return x_pad, y_pad, sents


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def batch_iter_bert(x, y, sents, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        text = sents[i]
	if not text:
	    text = 'a'
        yield [bert_encode(sents[i]) for i in x_shuffle[start_id:end_id]], y_shuffle[start_id:end_id]



