from tagger import cleanup
import numpy as np
import pandas as pd

from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Bidirectional, TimeDistributed
from keras_contrib.layers import CRF
from sys import stdout
from tqdm import tqdm


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
                input_length=params['max_utt_len']))
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

    def train(self, data):
        print('Training network', flush=True)
        train = data.sets['train']
        target = data.sets['target']
        self.model.fit(
            train,
            target,
            batch_size=data.params['batch_size'],
            validation_split=data.params['validation_split'],
            epochs=data.params['num_epochs'])

    def test(self, data, tagger):
        print('Evaluating test set performance...', flush=True)
        hash_test = data.sets['hash_test'].reset_index()
        predictions = []
        for sentence in tqdm(
                data.sets['utt_test']['frase'], ascii=True, file=stdout):
            prediction = self.predict(sentence, data.params, tagger.tokenizer)
            predictions.append(prediction)
        predictions = pd.DataFrame(predictions)
        predictions.columns = ['pred']
        total_utts = hash_test.shape[0]
        hash_test = data.sets['hash_test'].reset_index()
        positives = 0
        for i in range(total_utts):
            if predictions.iloc[i][0] == hash_test.iloc[i]['tag']:
                positives += 1
        acc = (positives / total_utts)
        print('Accuracy: {:.4f}'.format(acc))
        return acc

    def predict(self, sentence, params, tokenizer):
        # Tratamieto
        sentence_len = len(sentence.strip().split())
        tratada = cleanup(sentence)
        test = pad_sequences(
            tokenizer.texts_to_sequences(np.array([tratada])),
            maxlen=params['max_utt_len'],
            padding='post')
        predict = self.model.predict(test)
        traduccion = []
        for i in range(0, sentence_len):
            prediction = np.argmax(predict[0][i])
            traduccion += [params['learn_tags'][prediction]]
        traduccion = " ".join(str(x) for x in traduccion)
        return traduccion

    def save(self, nn_filename):
        print('Saving trained network to {}'.format(nn_filename))
        self.model.save(nn_filename)

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

    def load(self, nn_filename, params):
        """Loads a presaved H5 network and the tokenizer used to encode
        the words"""
        load_crf = params['CRF']
        if load_crf is True:
            self.model = load_model(
                nn_filename,
                custom_objects=self.create_custom_objects(),
                compile=True)
        else:
            self.model = load_model(nn_filename, compile=True)
        return self
