from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml
import pprint
import numpy as np
import pandas as pd
from tagger import cleanup

from sklearn.model_selection import train_test_split


class Data:

    params = dict()
    sets = dict()

    def __init__(self):
        pass

    def init(self, params_fname=None):
        filename = "params.yaml" if params_fname is None else params_fname
        print('Loading parameters from: {}'.format(filename))
        with open(filename, 'r') as ymlfile:
            self.params = yaml.load(ymlfile)
        np.random.seed(1337)
        self.params['learn_tags'] = list(self.params['amr'].keys())
        self.params['num_tags'] = len(self.params['learn_tags'])
        return self.prepare()

    def prepare(self):
        """Prepare the input datasets, and the vocabulary"""
        print('Reading input datasets', flush=True)
        datos = pd.read_csv(
            self.params['input_filename'], sep=',', encoding='utf-8')

        # Seleccionamos las columnas
        tag_header = self.params['tag_header']
        amr_header = self.params['amr_header']
        datos = datos[['frase', tag_header, amr_header]]
        datos.columns = ['frase', 'tag', 'amr']

        # Aplicamos correcciones a las frases:
        print('Cleaning up and splitting datasets', flush=True)
        datos['frase'] = datos['frase'].apply(lambda x: cleanup(x))
        datos['tag'] = datos['tag'].apply(lambda x: cleanup(x))
        U_dev, U_tst, H_dev, H_tst, A_dev, A_tst = train_test_split(
            datos[['frase']],
            datos[['tag']],
            datos[['amr']],
            test_size=self.params['test_size'],
            random_state=50)

        U_dev['frase'] = U_dev['frase'].apply(
            lambda x: x.replace('unk', self.params['UNK']))
        U_tst['frase'] = U_tst['frase'].apply(
            lambda x: x.replace('unk', self.params['UNK']))

        self.sets['utt_train'] = U_dev
        self.sets['utt_test'] = U_tst
        self.sets['hash_train'] = H_dev
        self.sets['hash_test'] = H_tst
        self.sets['amr_train'] = A_dev
        self.sets['amr_test'] = A_tst

        return self

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
