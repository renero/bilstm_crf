"""
Bi-LSTM with CRF

Usage:
    bilstm_crf (train | test | predict) <params>
    bilstm_crf [-h | --help]

Options:
  -h, --help    Show this message
"""

import sys
from docopt import docopt

from data import Data
from model import Model
from tagger import Tagger

# sys.argv = ["bilstm_crf.py", "test", "params_conditionings.yaml"]

if __name__ == '__main__':
    arguments = docopt(__doc__, version='0.1')
if arguments['<params>'] is None:
    print(__doc__)
    sys.exit(1)

data = Data().init(arguments['<params>'])
tagger = Tagger()
model = Model().init(data.params)

if arguments['train'] is True:
    tagger.init(data)
    model.train(data)
    model.save(data.params['def_nn_name'])
    tagger.save(data.params['def_tokenizer_name'])
elif arguments['test'] is True:
    tagger.load(data.params['def_tokenizer_name'])
    model.load(data.params['def_nn_name'], data.params)
    model.test(data, tagger)
else:
    model.load(data.params['def_nn_name'], data.params)
    sentence = input('Enter command: ')
    tagging = model.predict(sentence, data.params)
    print(
        'sentence: {}\ntagging.: {}\namr.....: {}'.format(
            sentence, tagging, tagger.expand_amr(tagging, data.params)),
        flush=True)
