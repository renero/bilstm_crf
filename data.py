from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml
import pprint
import numpy as np
import pandas as pd
import pickle
import vocabulary

from os.path import join
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def init():
    with open("params.yaml", 'r') as ymlfile:
        params = yaml.load(ymlfile)
    np.random.seed(1337)
    return params


def encode_input(dataset_name, datasets, params):
    """Simplified, Keras-way, creation of vocabulary, saving it
    padding and encoding."""
    tokenizer = Tokenizer(
        lower=True,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        oov_token=params['_UNK'])
    tokenizer.fit_on_texts(datasets[dataset_name]['frase'])
    encoded_utts = tokenizer.texts_to_sequences(
        datasets[dataset_name]['frase'])
    padded_encodings = pad_sequences(
        encoded_utts, maxlen=params['largo_max'], padding='post')
    params['vocabulary_size'] = len(tokenizer.word_index) + 1
    datasets['vocabulary'] = np.array(list(tokenizer.index_word.values()))

    # saving
    vocabulary.save(tokenizer, params)
    return padded_encodings


def build_response_input(dataset_name, data, params):
    """Builds the responde in a typical supervised learning problem
    from the input dataset."""
    data[dataset_name]['tag'] = data[dataset_name]['tag'].apply(
        lambda x: vocabulary.cleanup(x))
    target_lista = []
    for line in data[dataset_name]['tag']:
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

        while len(fila_lista) < params['largo_max']:
            fila_lista.append([1, 0, 0, 0, 0, 0])
        target_lista += [fila_lista]
    return np.array(target_lista)


def training_sets(datasets, params):
    """Simplified Keras version of the encoder"""
    train = encode_input('utt_train', datasets, params)
    target = build_response_input('hash_train', datasets, params)
    datasets['train'] = train
    datasets['target'] = target
    return datasets


def prepare(params):
    """Prepare the input datasets, and the vocabulary"""
    datos = pd.read_csv(
        join(join(params['working_path'], 'input'), params['input_filename']),
        sep=',',
        encoding='utf-8')

    # Seleccionamos las columnas
    datos = datos[['frase', 'tag_objeto', 'amr_objeto']]
    datos.columns = ['frase', 'tag', 'amr']

    # Aplicamos correcciones a las frases:
    datos['frase'] = datos['frase'].apply(lambda x: vocabulary.cleanup(x))
    datos['tag'] = datos['tag'].apply(lambda x: vocabulary.cleanup(x))
    # frases = datos[['frase']]
    U_dev, U_tst, H_dev, H_tst, A_dev, A_tst = train_test_split(
        datos[['frase']],
        datos[['tag']],
        datos[['amr']],
        test_size=params['test_size'],
        random_state=50)

    # Cambiar unk por _UNK
    U_dev['frase'] = U_dev['frase'].apply(lambda x: x.replace('unk', '_UNK'))
    U_tst['frase'] = U_tst['frase'].apply(lambda x: x.replace('unk', '_UNK'))

    datasets = dict()
    datasets['utt_train'] = U_dev
    datasets['utt_test'] = U_tst
    datasets['hash_train'] = H_dev
    datasets['hash_test'] = H_tst
    datasets['amr_train'] = A_dev
    datasets['amr_test'] = A_tst

    datasets = training_sets(datasets, params)
    return datasets


def save(datasets, params):
    with open('datasets.pickle', 'wb') as handle:
        pickle.dump(datasets, handle, protocol=pickle.HIGHEST_PROTOCOL)


def read(params):
    with open('datasets.pickle', 'rb') as handle:
        return pickle.load(handle)


def info(datasets, params):
    len_total = len(datasets['utt_train']) + len(datasets['utt_test'])
    print("Dataset total vol.", len_total)
    print('Training vols. 70% {:.0f}: (utts/resps): {:d}/{:d}'.format(
        len_total * 0.7, len(datasets['utt_train']),
        len(datasets['hash_train'])))
    print('Test data vols. 30% {:.0f}: (utts/resps): {:d}/{:d}'.format(
        len_total * 0.3, len(datasets['utt_test']),
        len(datasets['hash_test'])))
    print('Training dataset lengths (utterances/responses): {:d}/{:d}'.format(
        len(params['train']), len(params['target'])))


def info_sentence(datasets, index):
    print('Printout of instance #{:d} in dataset'.format(index))
    print('U> \'{}\''.format(datasets['utt_train'].iloc[index]['frase']))
    print('H> \'{}\''.format(datasets['hash_train'].iloc[index]['tag']))
    pprint.pprint(datasets['train'][index])
