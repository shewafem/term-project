import gensim
from gensim.models import Word2Vec
import pandas as pd
import re
import codecs
import numpy as np
patterns = "[!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"

w2v_model = Word2Vec.load("models/model_vkr.model")
array = []
array1 = []
for j in range(1, 21):
    result = 0
    words = 0
    result1 = 0
    with open(f"clean_result/vkr/text_{j}.txt", 'r', encoding="utf-8") as file:
        for line in file:
            line = re.sub(patterns, ' ', line)
            for word in line.split():
                if w2v_model.wv.has_index_for(word):
                    result += w2v_model.wv.get_vector(word).sum()
                    result1 += w2v_model.wv.get_vector(word)
                words += 1
    print(result/words, j)
    array.append(result/words)
    array1.append(result1/words)

sorted_array = sorted(array1, key=np.linalg.norm)

mid_index = len(sorted_array) // 2
first_half = sorted_array[:mid_index]
category1_vector = np.mean(first_half, axis=0)
second_half = sorted_array[mid_index:]
category2_vector = np.mean(second_half, axis=0)

print("Категория 1:", category1_vector.sum())
print("Категория 2:", category2_vector.sum())

print(w2v_model.wv.similar_by_vector(vector=category1_vector, topn=5))
print(w2v_model.wv.similar_by_vector(vector=category2_vector, topn=5))