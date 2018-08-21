"""
Bi-LSTM with CRF

Usage:
    bilstm_crf.py (train | test | predict) PARAMETERS_FILENAME
    bilstm_crf.py [-h | --help]

Options:
  -h, --help    Show this message
"""

import tagger

from data import Data
from docopt import docopt
from model import Model

if __name__ == '__main__':
    arguments = docopt(__doc__, version='FIXME')

data = Data(arguments['PARAMETERS_FILENAME'])
params = data.params
model = Model().init(params)

if arguments['train'] is True:
    datasets = data.prepare()
    model.train(datasets['train'], datasets['target'], params)
    model.save(params['def_nn_name'])
    data.save_tokenizer(params['def_tokenizer_name'])
elif arguments['test'] is True:
    datasets = data.prepare(load_tokenizer=True)
    model.load(params['def_nn_name'], params['def_tokenizer_name'], params)
    model.test(datasets, params)
else:
    model.load(params['def_nn_name'], params['def_tokenizer_name'], params)
    if 'SENTENCE' not in arguments:
        sentence = input('Enter command: ')
    else:
        sentence = arguments['SENTENCE']
    tagging = model.predict(sentence, params)
    print(
        'sentence: {}\ntagging.: {}\namr.....: {}'.format(
            sentence, tagging, tagger.expand_amr(tagging, params)),
        flush=True)
