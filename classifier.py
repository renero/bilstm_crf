"""
Classfier with three Bi-LSTM with CRF

Usage: classifier.py [-p PARAMS] [-h | --help]

Options:
  -h, --help    Show this message
  -p PARAMS     Use the FILE as input parameters. If not specified,
                classifier.yaml is used.
"""

import sys
import yaml

from docopt import docopt
from data import Data
from model import Model
from tagger import Tagger, expand_amr

# Read arguments if any.
sys.argv = ['classifier.py', '-p', 'classifier.yaml']
if __name__ == '__main__':
    arguments = docopt(__doc__, version='0.1')

# Read params file
filename = "classifier.yaml" if arguments['-p'] is None else arguments['-p']
with open(filename, 'r') as ymlfile:
    params = yaml.load(ymlfile)

# Init dictionaries that will contain networks and predictions.
data = dict()
amr = dict()
model = dict()
tagging = dict()
amr_exp = dict()

# Load all network weights
for nn_name in params['conf_files'].keys():
    print('> Loading network {}'.format(nn_name))
    network_params_file = params['conf_files'][nn_name]
    data[nn_name] = Data().init(network_params_file, load_dataset=False)
    amr[nn_name] = Tagger().load(data[nn_name].params['def_tokenizer_name'])
    model[nn_name] = Model().load(data[nn_name].params['def_nn_name'],
                                  data[nn_name].params)

sentence = input('Enter command: ')
print('sentence....:', sentence)
for nn_name in params['conf_files'].keys():
    print('  {}'.format(nn_name))
    tagging[nn_name] = model[nn_name].predict(sentence, data[nn_name].params,
                                              amr[nn_name].tokenizer)
    amr_exp[nn_name] = expand_amr(tagging[nn_name], data[nn_name].params)
    print(
        '    tagging...: {}\n    amr.......: {}'.format(
            tagging[nn_name], amr_exp[nn_name]),
        flush=True)
