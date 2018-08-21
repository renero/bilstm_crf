"""
bilstm_crf

Usage:
    bilstm_crf train
    bilstm_crf test
    bilstm_crf predict SENTENCE
    bilstm_crf [-h | --help]

Options:
  -h, --help    Show this message
"""

import data
import pandas as pd

from docopt import docopt, DocoptExit
from model import Model
from tqdm import tqdm
from sys import stdout

if __name__ == '__main__':
    try:
        arguments = docopt(__doc__, version='FIXME')
    except DocoptExit:
        print('No valid arguments to docopt(), taking default behavior.')
        arguments = dict()
        arguments['train'] = False
        arguments['test'] = False
        arguments['predict'] = True

params = data.init()
model = Model().init(params)

if arguments['train'] is True:
    datasets = data.prepare(params)
    model.train(datasets['train'], datasets['target'], params)
    model.save(params['def_nn_name'])
elif arguments['test'] is True:
    datasets = data.prepare(params)
    model.load(params['def_nn_name'], params['CRF'])
    print('Evaluating test set performance...', flush=True)
    frases_test = datasets['utt_test'].reset_index()
    hash_test = datasets['hash_test'].reset_index()
    amr_test = datasets['amr_test'].reset_index()
    pred = []
    for sentence in tqdm(
            datasets['utt_test']['frase'], ascii=True, file=stdout):
        traduccion = model.predict(sentence, params)
        pred.append(traduccion)
    pred = pd.DataFrame(pred)
    pred.columns = ['pred']
    total_utts = hash_test.shape[0]
    hash_test = datasets['hash_test'].reset_index()
    positives = 0
    for i in range(total_utts):
        if pred.iloc[i][0] == hash_test.iloc[i]['tag']:
            positives += 1
    print('Accuracy: {:.4f}'.format(positives / total_utts))
else:
    model.load(params['def_nn_name'], params)
    if arguments['SENTENCE'] is None:
        sentence = input('Enter command: ')
    else:
        sentence = arguments['SENTENCE']
    tagging = model.predict(sentence, params)
    print('sentence: {}\ntagging.: {}'.format(sentence, tagging), flush=True)
