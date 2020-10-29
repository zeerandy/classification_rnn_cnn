from collections import Counter
from tensorflow import keras as kr

list_data_1 = "hello word i am wang wei i am wang wei  word i am wang wei"
list_data_2 = "对不起，我是警察，，，，，，三年又三年。我想做个好人，我是警察"
list_word = ["对", "不", "起", "我", "是", "警", "察"]
contents = ["我不是警察吗"]
data_id = []


counter1 = Counter(list_data_1)
counter2 = Counter(list_data_2)

list_1_1 = counter1.most_common(4)
list_2_1 = counter2.most_common(4)

words, _ = list(zip(*list_1_1))
words_1, _ = zip(*list_1_1)
words_1 = list(words_1)


words = ['<PAD>'] + list(words)
word_to_id = dict(zip(list_word, range(len(list_word))))

for i in range(len(contents)):
    data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])

x_pad = kr.preprocessing.sequence.pad_sequences(data_id, 10)



print(counter1)
print(counter2)
print(list_1_1)
print(list_2_1)

print(words)
print(words_1)
print(word_to_id)
print(data_id)
print(x_pad)