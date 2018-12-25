from gensim.models import KeyedVectors
import numpy as np

EMBEDDING_FILE = '/root/shen-sz/google300/GoogleNews-vectors-negative300.bin'
word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
with open('vocab.txt', 'r') as f, open('vocab_vec.txt', 'w') as w:
	for line in f.readlines():
		word = line.strip('\n')
		word = word.strip()
		if word in word2vec.vocab:
			vec = word2vec.word_vec(word)
		else:
			vec = np.random.rand(300) * 2 - 1
		w.write(word + ' ')
		w.write(' '.join([str(x) for x in list(vec)]) + '\n')
	
		


