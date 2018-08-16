from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.layers import Bidirectional, TimeDistributed


class Model:

    model = Sequential()

    def __init__(self, params):
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
            train, target, batch_size=params['batch_size'],
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
