import gensim
from gensim.models import Word2Vec
import pandas as pd
import re

#Определяем нежелательные символы
patterns = "[!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"

response = []

#Обрабатываем файл
with open('resources/med.tsv', encoding='utf-8') as f:
    lines = f.readlines()
    columns = lines[0].split('\t')
    for line in lines[1:]:
        temp = line.split('\t')
        response.append(re.sub(patterns, ' ', temp[0]))

# Создаем DataFrame для хранения обработанных текстов.
data = pd.DataFrame(list(zip(response)))
data.columns = ['response']
response_base = data.response.apply(gensim.utils.simple_preprocess)

# Создаем модель Word2Vec с указанными параметрами:
# - sentences: токенизированный текст.
# - min_count: минимальное количество вхождений слова для его учета в модели.
# - window: размер окна (контекста) для каждого слова.
# - vector_size: размерность вектора (количество признаков).
# - alpha: скорость обучения.
# - negative: количество негативных примеров для обучения.
# - min_alpha: минимальное значение скорости обучения.
# - sample: порог частоты слов для их случайного отбрасывания.
model = Word2Vec(
    sentences=response_base,
    min_count=10,
    window=2,
    vector_size=16,
    alpha=0.03,
    negative=15,
    min_alpha=0.0007,
    sample=6e-5
)

# Обучение модели
model.build_vocab(response_base, update=True)
model.train(response_base, total_examples=model.corpus_count, epochs=model.epochs)


print(model.corpus_count)
print(model.wv.has_index_for("diabetes"))

print(model.wv.similar_by_vector(model.wv['diabetes']))

model.save("resources/model_med.model")

print(model.corpus_count)

# Определение, какое из слов "blood", "insulin" наиболее похоже на слово "glucose".
print(model.wv.most_similar_to_given("glucose", ["blood", "insulin"]))
