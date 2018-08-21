from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml
import pprint
import numpy as np
import pandas as pd
import tagger

from keras.preprocessing.sequence import pad_sequences
from os.path import join
from sklearn.model_selection import train_test_split
from sys import stdout
from tqdm import tqdm


class Data:

    params = dict()
    tokenizer = None

    def __init__(self):
        self.init()

    def init(self):
        with open("params.yaml", 'r') as ymlfile:
            self.params = yaml.load(ymlfile)
        np.random.seed(1337)
        # return self

    def encode_utterances(self, dataset_name, datasets):
        """Simplified, Keras-way, creation of vocabulary, saving it
        padding and encoding."""
        # = Tokenizer(
        #     lower=True,
        #     filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        #     oov_token=self.params['_UNK'])
        # self.tokenizer.fit_on_texts(datasets[dataset_name]['frase'])
        encoded_utts = self.tokenizer.texts_to_sequences(
            datasets[dataset_name]['frase'])
        padded_encodings = pad_sequences(
            encoded_utts, maxlen=self.params['largo_max'], padding='post')
        self.params['vocabulary_size'] = len(self.tokenizer.word_index) + 1
        datasets['vocabulary'] = np.array(
            list(self.tokenizer.index_word.values()))
        return padded_encodings

    def encode_prediction(self, dataset_name, data):
        """Builds the responde in a typical supervised learning problem
        from the input dataset."""
        data[dataset_name]['tag'] = data[dataset_name]['tag'].apply(
            lambda x: tagger.cleanup(x))
        target_lista = []
        for line in tqdm(data[dataset_name]['tag'], ascii=True, file=stdout):
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

            while len(fila_lista) < self.params['largo_max']:
                fila_lista.append([1, 0, 0, 0, 0, 0])
            target_lista += [fila_lista]
        return np.array(target_lista)

    def encode(self, datasets):
        """Simplified Keras version of the encoder"""
        train = self.encode_utterances('utt_train', datasets)
        target = self.encode_prediction('hash_train', datasets)
        datasets['train'] = train
        datasets['target'] = target
        return datasets

    def prepare(self, load_tokenizer=False):
        """Prepare the input datasets, and the vocabulary"""
        print('Reading input datasets', flush=True)
        datos = pd.read_csv(
            join(
                join(self.params['working_path'], 'input'),
                self.params['input_filename']),
            sep=',',
            encoding='utf-8')

        # Seleccionamos las columnas
        datos = datos[['frase', 'tag_objeto', 'amr_objeto']]
        datos.columns = ['frase', 'tag', 'amr']

        # Aplicamos correcciones a las frases:
        print('Cleaning up and splitting datasets', flush=True)
        datos['frase'] = datos['frase'].apply(lambda x: tagger.cleanup(x))
        datos['tag'] = datos['tag'].apply(lambda x: tagger.cleanup(x))
        # frases = datos[['frase']]
        U_dev, U_tst, H_dev, H_tst, A_dev, A_tst = train_test_split(
            datos[['frase']],
            datos[['tag']],
            datos[['amr']],
            test_size=self.params['test_size'],
            random_state=50)

        U_dev['frase'] = U_dev['frase'].apply(
            lambda x: x.replace('unk', '_UNK'))
        U_tst['frase'] = U_tst['frase'].apply(
            lambda x: x.replace('unk', '_UNK'))

        datasets = dict()
        datasets['utt_train'] = U_dev
        datasets['utt_test'] = U_tst
        datasets['hash_train'] = H_dev
        datasets['hash_test'] = H_tst
        datasets['amr_train'] = A_dev
        datasets['amr_test'] = A_tst

        if load_tokenizer is False:
            self.tokenizer = tagger.init(datasets['utt_train']['frase'],
                                         self.params['_UNK'])
        else:
            self.load_tokenizer(self.params['def_tokenizer_name'])
        print('Encoding training sets.', flush=True)
        datasets = self.encode(datasets)
        return datasets

    def save_tokenizer(self, tokenizer_filename):
        tagger.save(self.tokenizer, tokenizer_filename)

    def load_tokenizer(self, tokenizer_filename):
        self.tokenizer = tagger.read(tokenizer_filename)

    def info(self, datasets):
        len_total = len(datasets['utt_train']) + len(datasets['utt_test'])
        print("Dataset total vol.", len_total)
        print('Training vols. 70% {:.0f}: (utts/resps): {:d}/{:d}'.format(
            len_total * 0.7, len(datasets['utt_train']),
            len(datasets['hash_train'])))
        print('Test data vols. 30% {:.0f}: (utts/resps): {:d}/{:d}'.format(
            len_total * 0.3, len(datasets['utt_test']),
            len(datasets['hash_test'])))
        print('Training dataset lengths (utts/resps): {:d}/{:d}'.format(
            len(self.params['train']), len(self.params['target'])))

    def info_sentence(self, datasets, index):
        print('Printout of instance #{:d} in dataset'.format(index))
        print('U> \'{}\''.format(datasets['utt_train'].iloc[index]['frase']))
        print('H> \'{}\''.format(datasets['hash_train'].iloc[index]['tag']))
        pprint.pprint(datasets['train'][index])
