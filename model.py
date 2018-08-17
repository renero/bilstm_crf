import vocabulary
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Bidirectional, TimeDistributed
from keras_contrib.layers import CRF
from keras_contrib.utils import save_load_utils


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
        crf = CRF(params['num_tags'])  # CRF layer
        self.model.add(crf)
        self.model.summary()
        self.model.compile(
            loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])
        # self.model.compile(
        # 'adam', 'categorical_crossentropy', metrics=['accuracy'])

    def train(self, train, target, params):
        print('Train...', flush=True)
        self.model.fit(
            train,
            target,
            batch_size=params['batch_size'],
            validation_split=params['validation_split'],
            epochs=params['num_epochs'])
        print('done.', flush=True)

    def save(self, filename=None, params=None):
        """Saves the network to a file"""
        if filename is None:
            if params is not None:
                self.model.save(params['default_nn_name'])
            else:
                self.model.save('./output/BI_LSTM_entities.h5')
        else:
            # self.model.save(filename)
            save_load_utils.save_all_weights(
                self.model, filename, include_optimizer=False)

    def load(self, filename):
        # self.model = load_model(filename)
        save_load_utils.load_all_weights(self.model, filename)

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
