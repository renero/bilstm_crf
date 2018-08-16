import vocabulary
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Bidirectional, TimeDistributed


class Model:

    model = Sequential()

    def __init__(self):
        pass

    def init(self, params):
        self.model.add(
            Embedding(
                params['vocabulary_size'],
                128,
                input_length=params['largo_max']))
        self.model.add(LSTM(64, return_sequences=True))
        self.model.add(Bidirectional(LSTM(128, return_sequences=True)))
        self.model.add(TimeDistributed(Dense(6, activation='sigmoid')))
        self.model.summary()
        self.model.compile(
            'adam', 'categorical_crossentropy', metrics=['accuracy'])

    def train(self, train, target, params):
        print('Train...', flush=True)
        self.model.fit(
            train,
            target,
            batch_size=params['batch_size'],
            epochs=params['num_epochs'])
        print('done.', flush=True)

    def save(self, name=None, params=None):
        """Saves the network to a file"""
        if name is None:
            if params is not None:
                self.model.save(params['default_nn_name'])
            else:
                self.model.save('./output/BI_LSTM_entities.h5')
        else:
            self.model.save(name)

    def load(self, model_name):
        self.model = load_model(model_name)

    def predict(self, sentence, params):
        # Tratamieto
        largo_real_frase = len(sentence.split())
        tratada = vocabulary.cleanup(sentence)
        tokenizer = vocabulary.read(params)
        test = pad_sequences(
            tokenizer.texts_to_sequences(np.array([tratada])),
            maxlen=params['largo_max'],
            padding='post')

        # Predicción
        predict = self.model.predict(test)
        traduccion = []
        for i in range(0, largo_real_frase):  # TODO
            if np.argmax(predict[0][i]) == 0:
                traduccion += ['x']
            if np.argmax(predict[0][i]) == 1:
                traduccion += ['ot']
            if np.argmax(predict[0][i]) == 2:
                traduccion += ['oc']
            if np.argmax(predict[0][i]) == 3:
                traduccion += ['oh']
            if np.argmax(predict[0][i]) == 4:
                traduccion += ['or']
            if np.argmax(predict[0][i]) == 5:
                traduccion += ['os']

        # Escritura en fichero:
        traduccion = " ".join(str(x) for x in traduccion)
        return traduccion
