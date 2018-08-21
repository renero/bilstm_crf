import vocabulary
import numpy as np

from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Bidirectional, TimeDistributed
from keras_contrib.layers import CRF


class Model:

    model = Sequential()
    _embedded_size = 128
    _bidirectional_size = 128
    _dense_size = 128
    _optimizer = 'adam'

    def __init__(self):
        pass

    def init(self, params):
        self.model.add(
            Embedding(
                params['vocabulary_size'],
                self._embedded_size,
                input_length=params['largo_max']))
        self.model.add(
            Bidirectional(
                LSTM(self._bidirectional_size, return_sequences=True)))
        self.model.add(Dense(self._dense_size, activation='tanh'))
        if params['CRF'] is True:
            crf = CRF(params['num_tags'], sparse_target=False)
            self.model.add(crf)
            loss = crf.loss_function
            metric = crf.accuracy
        else:
            self.model.add(
                TimeDistributed(
                    Dense(params['num_tags'], activation='softmax')))
            loss = 'categorical_crossentropy'
            metric = 'accuracy'
        self.model.summary()
        self.model.compile(
            loss=loss, optimizer=self._optimizer, metrics=[metric])
        return self

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
                self.model.save('./output/nn_entities.h5')
        else:
            self.model.save(filename)

    def create_custom_objects(self):
        instanceHolder = {"instance": None}

        class ClassWrapper(CRF):
            def __init__(self, *args, **kwargs):
                instanceHolder["instance"] = self
                super(ClassWrapper, self).__init__(*args, **kwargs)

        def loss(*args):
            method = getattr(instanceHolder["instance"], "loss_function")
            return method(*args)

        def accuracy(*args):
            method = getattr(instanceHolder["instance"], "accuracy")
            return method(*args)

        return {
            "ClassWrapper": ClassWrapper,
            "CRF": ClassWrapper,
            "loss": loss,
            "accuracy": accuracy
        }

    def load(self, filename, load_crf):
        if load_crf is True:
            self.model = load_model(
                filename,
                custom_objects=self.create_custom_objects(),
                compile=True)
        else:
            self.model = load_model(filename, compile=True)

    def predict(self, sentence, params):
        # Tratamieto
        largo_real_frase = len(sentence.split())
        tratada = vocabulary.cleanup(sentence)
        tokenizer = vocabulary.read(params['def_tokenizer_name'])
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
