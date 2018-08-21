from numpy import array
from pprint import pprint
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding  # define documents
from keras.utils import to_categorical


docs = [
    'Well done!', 'I have 3 pastels ??'
    'Good work', 'Great effort', 'nice work', 'Excellent!', 'Weak',
    'Poor effort!', 'not good', 'poor work', 'Could have done better.'
]
# define class labels
labels = array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
# integer encode the documents
vocab_size = 50
encoded_docs = [one_hot(d, vocab_size) for d in docs]
pprint(encoded_docs)
pprint(to_categorical(encoded_docs[8], num_classes=vocab_size))
# pad documents to a max length of 4 words
max_length = 6
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)

# define the model
model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
print(model.summary())

# fit the model
model.fit(padded_docs, labels, epochs=100, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy * 100))

# another approach

t = Tokenizer(
    lower=True,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    oov_token='_UNK')
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(docs)
print(encoded_docs)
# pad documents to a max length of 4 words
max_length = 6
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)
t.index_word.values()
r = t.texts_to_sequences(array(['I made a good work', 'I did effort']))
r
t.index_word[6]

# hola
# hola
# hola
# hola
# hola
# hola

# Sample of FAST encoding of tags
sents = ['x ot oh x x x x oc', 'x x x x oh oc x']
tag_size = 6
encoded_sent = [one_hot(d, tag_size) for d in sents]
pprint(encoded_sent)
[pprint(to_categorical(enc, num_classes=6)) for enc in encoded_sent]

# test on encoding with functional
# test on encoding with functional
# test on encoding with functional
# test on encoding with functional
# test on encoding with functional
# test on encoding with functional
# test on encoding with functional
# test on encoding with functional
# test on encoding with functional
# test on encoding with functional
# test on encoding with functional
# test on encoding with functional
# test on encoding with functional
# test on encoding with functional
# test on encoding with functional
# test on encoding with functional
# test on encoding with functional

# %%
import numpy as np
params = dict()
params['learn_tags'] = ['x', 'ot', 'oc', 'oh', 'or', 'os']
params['num_tags'] = 6
params['void_tag'] = 'x'
line = 'x x oc oc x'
tags_array = line.strip().split()

target = np.zeros([len(tags_array), len(params['learn_tags'])], dtype=np.int8)
for j, tag in enumerate(tags_array):
    one_pos = params['learn_tags'].index(tag)
    target[j][one_pos] = 1
target

# %%
fila_lista = []
for space_separated_fragment in line.strip().split():  # TODO
    if space_separated_fragment.strip().lower() == 'x':
        fila_lista += [[1, 0, 0, 0, 0, 0]]
    if space_separated_fragment.strip().lower() == 'ot':
        fila_lista += [[0, 1, 0, 0, 0, 0]]
    if space_separated_fragment.strip().lower() == 'oc':
        fila_lista += [[0, 0, 1, 0, 0, 0]]
    if space_separated_fragment.strip().lower() == 'oh':
        fila_lista += [[0, 0, 0, 1, 0, 0]]
    if space_separated_fragment.strip().lower() == 'or':
        fila_lista += [[0, 0, 0, 0, 1, 0]]
    if space_separated_fragment.strip().lower() == 'os':
        fila_lista += [[0, 0, 0, 0, 0, 1]]
fila_lista

# build the amr
# build the amr
# build the amr
# build the amr
# build the amr
# build the amr


# %%
line = 'x x oc ot x'
params['amr'] = {'x': 'O', 'ot': 'O-TARJETA', 'oc': 'O-COBERTURA',
                 'oh': 'O-CENTROS', 'or': 'O-REEMBOLSO', 'os': 'O-SEGURO'}
amr = ''
tags = line.strip().split()
valid_tags = params['learn_tags'].copy()
del valid_tags[params['learn_tags'].index(params['void_tag'])]
for valid_tag in valid_tags:
    if any(valid_tag in tag for tag in tags):
        amr += '{} '.format(params['amr'][valid_tag])
print(amr.strip())
