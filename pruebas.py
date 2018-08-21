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
